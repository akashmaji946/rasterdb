-- Test GPU hash outer joins.
-- Run with:
--   conda run -n rasterdf ./build/release/duckdb -unsigned < test/test_outer_join_gpu.sql

LOAD '/home/akashmaji/Device/IMPORTANT/rasterdb/build/release/extension/rasterdb/rasterdb.duckdb_extension';

CREATE TABLE oj_left_i32 (
    lid INTEGER,
    k INTEGER,
    lv INTEGER
);

CREATE TABLE oj_right_i32 (
    rid INTEGER,
    k INTEGER,
    rv INTEGER
);

INSERT INTO oj_left_i32 VALUES
    (1, 10, 100),
    (2, 20, 200),
    (3, 20, 201),
    (4, 30, 300),
    (5, 50, 500);

INSERT INTO oj_right_i32 VALUES
    (11, 20, 2000),
    (12, 20, 2001),
    (13, 40, 4000),
    (14, 50, 5000);

.print '=== OUTER JOIN I32: LEFT preserves unmatched left rows ==='
SELECT * FROM gpu_execution(
    'SELECT l.lid, l.k AS lk, l.lv, r.rid, r.k AS rk, r.rv
       FROM oj_left_i32 l LEFT JOIN oj_right_i32 r ON l.k = r.k
      ORDER BY l.lid, r.rid'
);

.print '=== OUTER JOIN I32: RIGHT preserves unmatched right rows ==='
SELECT * FROM gpu_execution(
    'SELECT l.lid, l.k AS lk, l.lv, r.rid, r.k AS rk, r.rv
       FROM oj_left_i32 l RIGHT JOIN oj_right_i32 r ON l.k = r.k
      ORDER BY r.rid, l.lid'
);

.print '=== OUTER JOIN I32: FULL preserves both sides ==='
SELECT * FROM gpu_execution(
    'SELECT l.lid, l.k AS lk, l.lv, r.rid, r.k AS rk, r.rv
       FROM oj_left_i32 l FULL OUTER JOIN oj_right_i32 r ON l.k = r.k
      ORDER BY l.lid, r.rid'
);

CREATE TABLE oj_left_i64 (
    lid INTEGER,
    k BIGINT,
    lv INTEGER
);

CREATE TABLE oj_right_i64 (
    rid INTEGER,
    k BIGINT,
    rv INTEGER
);

INSERT INTO oj_left_i64 VALUES
    (1, 10000000000, 10),
    (2, 20000000000, 20),
    (3, 30000000000, 30);

INSERT INTO oj_right_i64 VALUES
    (11, 20000000000, 200),
    (12, 40000000000, 400);

.print '=== OUTER JOIN I64: FULL preserves INT64 key rows ==='
SELECT * FROM gpu_execution(
    'SELECT l.lid, l.k AS lk, r.rid, r.k AS rk
       FROM oj_left_i64 l FULL OUTER JOIN oj_right_i64 r ON l.k = r.k
      ORDER BY l.lid, r.rid'
);

CREATE TABLE oj_left_d64 (
    lid INTEGER,
    k DECIMAL(18,2),
    lv INTEGER
);

CREATE TABLE oj_right_d64 (
    rid INTEGER,
    k DECIMAL(18,2),
    rv INTEGER
);

INSERT INTO oj_left_d64 VALUES
    (1, -900000000000.25, 10),
    (2, 0.01, 20),
    (3, 1234567890123456.78, 30);

INSERT INTO oj_right_d64 VALUES
    (11, 0.01, 200),
    (12, 999999999999.99, 999);

.print '=== OUTER JOIN D64: LEFT preserves DECIMAL64 unmatched rows ==='
SELECT * FROM gpu_execution(
    'SELECT l.lid, l.k AS lk, r.rid, r.k AS rk
       FROM oj_left_d64 l LEFT JOIN oj_right_d64 r ON l.k = r.k
      ORDER BY l.lid, r.rid'
);
