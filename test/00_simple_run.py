#!/usr/bin/env python3
"""
Run all 10 SPJ queries from 00_simple.sql against RasterDB engine.

This demonstrates the DuckDB (storage) + librasterdf (GPU compute) integration,
which is the RasterDB equivalent of Sirius's DuckDB + libcudf pipeline.

Usage:
    # First create the database:
    python test/00_simple_setup.py

    # Then run queries:
    python test/00_simple_run.py
"""

import os
import sys
import time

# Add paths
RASTERDB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RASTERDF_PYTHON = os.path.join(RASTERDB_ROOT, "..", "rasterdf", "python")
sys.path.insert(0, os.path.join(RASTERDB_ROOT, "scripts"))
sys.path.insert(0, RASTERDF_PYTHON)

from rasterdb_engine import RasterDBEngine

DB_PATH = os.path.join(RASTERDB_ROOT, "test", "00_simple.db")
SQL_PATH = os.path.join(RASTERDB_ROOT, "test", "00_simple.sql")


def printline(char="=", width=70):
    print(char * width)


def load_queries(sql_path):
    """Load SQL queries from file, skipping comments and blanks."""
    queries = []
    with open(sql_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            queries.append(line)
    return queries


def main():
    # Check database exists
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        print("Run: python test/00_simple_setup.py")
        sys.exit(1)

    # Load queries
    queries = load_queries(SQL_PATH)
    print(f"Loaded {len(queries)} queries from {SQL_PATH}\n")

    # Initialize RasterDB engine
    printline()
    print("RasterDB Engine — DuckDB + librasterdf (Vulkan GPU)")
    printline()
    print(f"Database: {DB_PATH}")
    print(f"RasterDF: {RASTERDF_PYTHON}")

    engine = RasterDBEngine(
        db_path=DB_PATH,
        rasterdf_path=RASTERDF_PYTHON,
        gpu_mem_mb=2048,
        debug=False,
    )
    print("GPU context initialized.\n")

    # Run each query
    total_gpu_time = 0.0
    passed = 0
    failed = 0

    for i, sql in enumerate(queries, 1):
        printline("-")
        print(f"Q{i}: {sql}")
        printline("-")

        try:
            t0 = time.perf_counter()
            result = engine.execute(sql)
            gpu_ms = (time.perf_counter() - t0) * 1000
            total_gpu_time += gpu_ms

            nrows = result.shape[0]
            ncols = result.shape[1]
            print(f"  Result: {nrows:,} rows x {ncols} cols  ({gpu_ms:.2f} ms)")

            # Show first few rows
            if nrows <= 20:
                print(result)
            else:
                print(result.head(10))

            passed += 1
            print(f"  [PASS]")

        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")
            failed += 1

        print()

    # Summary
    printline("=")
    print(f"RESULTS: {passed}/{len(queries)} passed, {failed} failed")
    print(f"Total GPU execution time: {total_gpu_time:.2f} ms")
    printline("=")
    sys.stdout.flush()

    engine.close()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
