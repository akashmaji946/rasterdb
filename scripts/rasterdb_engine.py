#!/usr/bin/env python3
"""
RasterDB Engine — GPU-accelerated SQL execution via librasterdf.

This is the Python bridge that replaces cuDF-based Sirius with RasterDF-based
RasterDB. It uses DuckDB for SQL storage/parsing and RasterDF for GPU execution.

Architecture (mirrors Sirius):
  DuckDB (storage + SQL parsing)
      |
      v
  RasterDB Engine (this file)
      |
      v
  librasterdf (Vulkan GPU compute)

Usage:
    engine = RasterDBEngine("my.db", rasterdf_path="/path/to/rasterdf/python")
    result = engine.execute("SELECT * FROM users WHERE id = 42")
    print(result)
"""

import re
import sys
import os
import ctypes
import numpy as np
import duckdb


class RasterDBEngine:
    """GPU-accelerated SQL engine: DuckDB for storage, RasterDF for compute."""

    def __init__(self, db_path, rasterdf_path=None, gpu_mem_mb=2048, debug=False):
        """
        Args:
            db_path: Path to DuckDB database file.
            rasterdf_path: Path to rasterdf/python directory.
            gpu_mem_mb: GPU memory limit in MB for rasterdf.
            debug: Enable rasterdf debug output.
        """
        # Set shader directory env var so librasterdf finds shaders from any cwd
        if rasterdf_path:
            rasterdf_root = os.path.dirname(rasterdf_path)  # parent of python/
            shader_dir = os.path.join(rasterdf_root, "shaders", "compiled")
            if os.path.isdir(shader_dir):
                os.environ["RASTERDF_SHADER_DIR"] = shader_dir

        # Import rasterdf
        if rasterdf_path:
            sys.path.insert(0, rasterdf_path)
        import rasterdf as rdf
        self.rdf = rdf

        # DuckDB connection (read-only for queries)
        self.db_path = db_path
        self.con = duckdb.connect(db_path, read_only=True)

        # Table cache: table_name -> rdf.DataFrame
        self._cache = {}

        # Init rasterdf GPU context
        rdf.init_device("default")
        rdf.init_context_memory(limit_mb=gpu_mem_mb, heap_ratio=0.9)
        rdf.set_debug_mode(debug)

    def close(self):
        """Release resources."""
        self._cache.clear()
        self.con.close()
        self.rdf.free_context()

    # ------------------------------------------------------------------
    # Table loading: DuckDB -> rasterdf DataFrame (GPU)
    # ------------------------------------------------------------------

    def _load_table(self, table_name):
        """Load a DuckDB table into a GPU-resident rasterdf DataFrame."""
        if table_name in self._cache:
            return self._cache[table_name]

        result = self.con.execute(f"SELECT * FROM {table_name}").fetchnumpy()
        col_names = list(result.keys())

        data = {}
        for name in col_names:
            arr = result[name]
            # Convert to int32 (rasterdf's primary type for now)
            data[name] = arr.astype(np.int32)

        df = self.rdf.DataFrame(data)
        self._cache[table_name] = df
        return df

    def invalidate_cache(self, table_name=None):
        """Clear cached GPU tables (e.g., after INSERT)."""
        if table_name:
            self._cache.pop(table_name, None)
        else:
            self._cache.clear()

    # ------------------------------------------------------------------
    # SQL Parser — decomposes simple SPJ queries into clause components
    # ------------------------------------------------------------------

    def _parse_sql(self, sql):
        """Parse a simple SQL query into execution plan components."""
        sql = re.sub(r'\s+', ' ', sql.strip().rstrip(';').strip())
        upper = sql.upper()

        plan = {
            'select': '*',
            'tables': [],       # [(name, alias), ...]
            'join': None,       # {'table': str, 'alias': str, 'on_left': str, 'on_right': str}
            'where': None,      # raw WHERE string
            'group_by': None,   # [col, ...]
            'aggregates': {},   # {col_name: agg_func}
            'select_cols': [],  # parsed column list
        }

        # --- Extract GROUP BY (must be done before WHERE extraction) ---
        gb_match = re.search(r'\bGROUP\s+BY\s+(.+?)$', sql, re.I)
        if gb_match:
            plan['group_by'] = [c.strip() for c in gb_match.group(1).split(',')]
            sql = sql[:gb_match.start()].strip()
            upper = sql.upper()

        # --- Extract WHERE ---
        where_match = re.search(r'\bWHERE\s+(.+?)$', sql, re.I)
        if where_match:
            plan['where'] = where_match.group(1).strip()
            sql = sql[:where_match.start()].strip()
            upper = sql.upper()

        # --- Extract JOIN ... ON ... ---
        join_match = re.search(
            r'\b(?:INNER\s+)?JOIN\s+(\w+)(?:\s+(\w+))?\s+ON\s+(.+?)$',
            sql, re.I
        )
        if join_match:
            jt = join_match.group(1)
            ja = join_match.group(2)
            on_clause = join_match.group(3).strip()
            # Parse ON clause: a.col = b.col
            on_parts = re.split(r'\s*=\s*', on_clause)
            left_on = on_parts[0].split('.')[-1].strip() if '.' in on_parts[0] else on_parts[0].strip()
            right_on = on_parts[1].split('.')[-1].strip() if '.' in on_parts[1] else on_parts[1].strip()
            plan['join'] = {
                'table': jt,
                'alias': ja,
                'left_on': left_on,
                'right_on': right_on,
            }
            sql = sql[:join_match.start()].strip()
            upper = sql.upper()

        # --- Extract FROM ---
        from_match = re.search(r'\bFROM\s+(\w+)(?:\s+(\w+))?\s*$', sql, re.I)
        if from_match:
            tn = from_match.group(1)
            ta = from_match.group(2)
            plan['tables'].append((tn, ta))
            sql = sql[:from_match.start()].strip()
            upper = sql.upper()

        # --- Extract SELECT columns ---
        sel_match = re.match(r'^SELECT\s+(.+)$', sql, re.I)
        if sel_match:
            plan['select'] = sel_match.group(1).strip()

        # Parse select list for aggregates and column names
        plan['select_cols'], plan['aggregates'] = self._parse_select_list(plan['select'])

        return plan

    def _parse_select_list(self, select_str):
        """Parse SELECT clause, detecting aggregate functions."""
        if select_str.strip() == '*':
            return ['*'], {}

        cols = []
        aggs = {}
        for item in select_str.split(','):
            item = item.strip()
            agg_match = re.match(
                r'(SUM|COUNT|MIN|MAX|AVG|MEAN)\s*\(\s*(\*|\w+)\s*\)',
                item, re.I
            )
            if agg_match:
                func = agg_match.group(1).upper()
                col = agg_match.group(2)
                aggs[col] = func.lower()
                cols.append(f'{func}({col})')
            else:
                # Strip table alias prefix if present: u.col -> col
                col = item.split('.')[-1] if '.' in item else item
                cols.append(col)
        return cols, aggs

    # ------------------------------------------------------------------
    # WHERE clause evaluator — converts SQL predicates to rasterdf masks
    # ------------------------------------------------------------------

    def _eval_where(self, df, where_str, aliases=None):
        """Recursively evaluate a WHERE clause into a rasterdf boolean mask."""
        where_str = where_str.strip()

        # Remove matching outer parentheses
        if where_str.startswith('(') and where_str.endswith(')'):
            depth = 0
            for i, c in enumerate(where_str):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(where_str) - 1:
                    break
            else:
                # All parens matched — strip them
                where_str = where_str[1:-1].strip()

        # Handle NOT
        not_match = re.match(r'^NOT\s+(.+)$', where_str, re.I)
        if not_match:
            inner = not_match.group(1).strip()
            return ~self._eval_where(df, inner, aliases)

        # Handle AND (top-level, respecting parens)
        parts = self._split_top_level(where_str, 'AND')
        if len(parts) > 1:
            masks = [self._eval_where(df, p, aliases) for p in parts]
            result = masks[0]
            for m in masks[1:]:
                result = result & m
            return result

        # Handle OR (top-level, respecting parens)
        parts = self._split_top_level(where_str, 'OR')
        if len(parts) > 1:
            masks = [self._eval_where(df, p, aliases) for p in parts]
            result = masks[0]
            for m in masks[1:]:
                result = result | m
            return result

        # Handle comparison: col OP val_or_col
        for op_str, py_method in [('>=', '__ge__'), ('<=', '__le__'),
                                   ('!=', '__ne__'), ('<>', '__ne__'),
                                   ('>', '__gt__'), ('<', '__lt__'),
                                   ('=', '__eq__')]:
            if op_str in where_str:
                idx = where_str.index(op_str)
                left = where_str[:idx].strip()
                right = where_str[idx + len(op_str):].strip()

                col_name = self._resolve_col(left, aliases)
                col_series = df[col_name]

                # Right side: integer literal or column reference?
                try:
                    val = int(right)
                    return getattr(col_series, py_method)(val)
                except ValueError:
                    right_col = self._resolve_col(right, aliases)
                    return getattr(col_series, py_method)(df[right_col])

        raise ValueError(f"Cannot parse WHERE condition: {where_str}")

    def _resolve_col(self, ref, aliases=None):
        """Resolve 'alias.col' or 'table.col' to just 'col'."""
        if '.' in ref:
            return ref.split('.')[-1].strip()
        return ref.strip()

    def _split_top_level(self, expr, keyword):
        """Split expression at top-level keyword (AND/OR), respecting parens."""
        parts = []
        depth = 0
        current_tokens = []
        tokens = expr.split()

        for token in tokens:
            depth += token.count('(') - token.count(')')
            if depth == 0 and token.upper() == keyword:
                parts.append(' '.join(current_tokens))
                current_tokens = []
            else:
                current_tokens.append(token)

        if current_tokens:
            parts.append(' '.join(current_tokens))

        return parts if len(parts) > 1 else [expr]

    # ------------------------------------------------------------------
    # Query executor — runs parsed plan on GPU via rasterdf
    # ------------------------------------------------------------------

    def execute(self, sql):
        """Execute a SQL query on GPU, return rasterdf DataFrame result."""
        plan = self._parse_sql(sql)
        return self._execute_plan(plan)

    def execute_and_verify(self, sql):
        """Execute on GPU and verify against DuckDB CPU result."""
        gpu_result = self.execute(sql)
        cpu_result = self.con.execute(sql.rstrip(';')).fetchdf()
        return gpu_result, cpu_result

    def _execute_plan(self, plan):
        """Execute a parsed query plan on GPU."""
        # 1. Load primary table
        if not plan['tables']:
            raise ValueError("No FROM table found")
        primary_table, primary_alias = plan['tables'][0]
        df = self._load_table(primary_table)

        # 2. Handle JOIN
        if plan['join']:
            join_info = plan['join']
            right_df = self._load_table(join_info['table'])
            # Use the common join key
            on_col = join_info['left_on']
            df = df.merge(right_df, how='inner', on=on_col)

        # Build alias map for resolving column references
        aliases = {}
        if primary_alias:
            aliases[primary_alias] = primary_table
        if plan['join'] and plan['join']['alias']:
            aliases[plan['join']['alias']] = plan['join']['table']

        # 3. Apply WHERE filter
        if plan['where']:
            mask = self._eval_where(df, plan['where'], aliases)
            df = df[mask]

        # 4. Apply GROUP BY + aggregation
        if plan['group_by']:
            key_col = plan['group_by'][0]
            if plan['aggregates']:
                # Get the value column and aggregation function
                for val_col, agg_func in plan['aggregates'].items():
                    if val_col == '*':
                        # COUNT(*) — use the first non-key column
                        all_cols = df.columns
                        val_col = [c for c in all_cols if c != key_col][0]
                    gb = df.groupby(key_col)[val_col]
                    if agg_func == 'sum':
                        df = gb.sum()
                    elif agg_func == 'count':
                        df = gb.count()
                    elif agg_func == 'min':
                        df = gb.min()
                    elif agg_func == 'max':
                        df = gb.max()
                    elif agg_func in ('avg', 'mean'):
                        df = gb.mean()
                    else:
                        raise ValueError(f"Unsupported aggregate: {agg_func}")
                    break  # Only one aggregate supported per query for now
            else:
                raise ValueError("GROUP BY without aggregate not yet supported")

        # 5. Apply projection (SELECT columns)
        if plan['select_cols'] != ['*'] and not plan['group_by']:
            proj_cols = [self._resolve_col(c) for c in plan['select_cols']
                        if not re.match(r'(SUM|COUNT|MIN|MAX|AVG)\s*\(', c, re.I)]
            if proj_cols:
                df = df[proj_cols]

        return df
