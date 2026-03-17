#!/usr/bin/env python3
"""
Create 00_simple.db with test tables for RasterDB SPJ query testing.

Tables:
  - users(id INTEGER, age INTEGER, score INTEGER, dept INTEGER)  -- 1M rows
  - departments(dept INTEGER, budget INTEGER)                    -- 10 rows
"""

import os
import sys
import numpy as np
import duckdb

DB_PATH = os.path.join(os.path.dirname(__file__), "00_simple.db")


def main():
    # Remove existing DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed old {DB_PATH}")

    con = duckdb.connect(DB_PATH)

    # ---- users table (1M rows) ----
    np.random.seed(42)
    N = 1_000_000

    con.execute("""
        CREATE TABLE users (
            id    INTEGER NOT NULL,
            age   INTEGER NOT NULL,
            score INTEGER NOT NULL,
            dept  INTEGER NOT NULL
        )
    """)

    ids    = np.arange(1, N + 1, dtype=np.int32)
    ages   = np.random.randint(18, 70, size=N, dtype=np.int32)
    scores = np.random.randint(0, 100, size=N, dtype=np.int32)
    depts  = np.random.randint(1, 11, size=N, dtype=np.int32)

    con.execute(
        "INSERT INTO users SELECT unnest($1), unnest($2), unnest($3), unnest($4)",
        [ids, ages, scores, depts],
    )

    row_count = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"Created 'users' table: {row_count:,} rows")

    # ---- departments table (10 rows) ----
    con.execute("""
        CREATE TABLE departments (
            dept   INTEGER NOT NULL,
            budget INTEGER NOT NULL
        )
    """)

    dept_data = list(zip(
        range(1, 11),
        [100, 200, 150, 300, 250, 180, 220, 170, 310, 190],
    ))
    con.executemany("INSERT INTO departments VALUES (?, ?)", dept_data)

    row_count = con.execute("SELECT COUNT(*) FROM departments").fetchone()[0]
    print(f"Created 'departments' table: {row_count} rows")

    con.close()
    print(f"\nDatabase saved: {DB_PATH}")


if __name__ == "__main__":
    main()
