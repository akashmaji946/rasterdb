-- Test exact DECIMAL32/DECIMAL64 GPU transport and comparisons end-to-end.
-- Run with:
--   ./build/release/duckdb -unsigned < test/test_decimal_gpu.sql
--
-- This test intentionally covers the first fixed-point implementation step:
-- raw decimal projection/readback and same-type constant filters. Decimal
-- arithmetic/rescaling and DECIMAL128 operators require later shader support.

LOAD '/home/akashmaji/Device/IMPORTANT/rasterdb/build/release/extension/rasterdb/rasterdb.duckdb_extension';

CREATE TABLE decimal_exact (
    id INTEGER,
    d16 DECIMAL(4,2),
    d32 DECIMAL(9,2),
    d64 DECIMAL(18,2)
);

INSERT INTO decimal_exact VALUES
    (1, -0.05, -1250.25, -900000000000.25),
    (2,  0.00,     0.01,               0.01),
    (3, 12.34, 98765.43, 1234567890123456.78),
    (4, -99.99, -999999.99, -9999999999999999.99),
    (5, 99.99, 9999999.99, 9999999999999999.99),
    (6, -10.50, -0.01, -0.01),
    (7, 10.50, 1250.25, 900000000000.25),
    (8, 0.01, -98765.43, -1234567890123456.78),
    (9, -1.25, 42.42, 42.42),
    (10, 1.25, -42.42, -42.42),
    (11, -50.00, 500000.00, -5000000000000000.00),
    (12, 50.00, -500000.00, 5000000000000000.00);

.print '=== DECIMAL projection/readback: INT16 storage widened to GPU DECIMAL32 ==='
SELECT * FROM gpu_execution(
    'SELECT id, d16, d32, d64 FROM decimal_exact ORDER BY id'
);

.print '=== DECIMAL(4,2) filter: scaled narrow values ==='
SELECT * FROM gpu_execution(
    'SELECT id, d16 FROM decimal_exact WHERE d16 >= -0.05 AND d16 <= 12.34 ORDER BY id'
);

.print '=== DECIMAL(9,2) filter: scaled INT32 values ==='
SELECT * FROM gpu_execution(
    'SELECT id, d32 FROM decimal_exact WHERE d32 >= -1250.25 AND d32 <= 1250.25 ORDER BY d32 ASC'
);

.print '=== DECIMAL(18,2) filter: scaled INT64 values ==='
SELECT * FROM gpu_execution(
    'SELECT id, d64 FROM decimal_exact WHERE d64 >= -900000000000.25 AND d64 <= 900000000000.25 ORDER BY d64 ASC'
);

.print '=== DECIMAL(4,2) ORDER BY ASC: signed DECIMAL32 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d16 FROM decimal_exact ORDER BY d16 ASC'
);

.print '=== DECIMAL(9,2) ORDER BY DESC: signed DECIMAL32 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d32 FROM decimal_exact ORDER BY d32 DESC'
);

.print '=== DECIMAL(9,2) ORDER BY ASC: signed DECIMAL32 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d32 FROM decimal_exact ORDER BY d32 ASC'
);

.print '=== DECIMAL(18,2) ORDER BY ASC: signed DECIMAL64 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d64 FROM decimal_exact ORDER BY d64 ASC'
);

.print '=== DECIMAL(18,2) ORDER BY DESC: signed DECIMAL64 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d64 FROM decimal_exact ORDER BY d64 DESC'
);

-- ============================================================================
-- DECIMAL JOIN tests
-- ============================================================================

CREATE TABLE decimal_join_left (
    id INTEGER,
    k16 DECIMAL(4,2),
    k32 DECIMAL(9,2),
    k64 DECIMAL(18,2),
    payload INTEGER
);

CREATE TABLE decimal_join_right (
    rid INTEGER,
    k16 DECIMAL(4,2),
    k32 DECIMAL(9,2),
    k64 DECIMAL(18,2),
    marker INTEGER
);

INSERT INTO decimal_join_left VALUES
    (1, -99.99, -999999.99, -9999999999999999.99, 100),
    (2, -10.50,   -1250.25,    -900000000000.25, 200),
    (3,   0.00,       0.01,                 0.01, 300),
    (4,  12.34,   98765.43,  1234567890123456.78, 400),
    (5,  99.99, 9999999.99,  9999999999999999.99, 500),
    (6,   1.25,     -42.42,               -42.42, 600),
    (7,  50.00, -500000.00,  5000000000000000.00, 700),
    (8, -50.00,  500000.00, -5000000000000000.00, 800);

INSERT INTO decimal_join_right VALUES
    (101, -99.99, -999999.99, -9999999999999999.99, 10),
    (102, -10.50,   -1250.25,    -900000000000.25, 20),
    (103,   0.00,       0.01,                 0.01, 30),
    (104,  12.34,   98765.43,  1234567890123456.78, 40),
    (105,  99.99, 9999999.99,  9999999999999999.99, 50),
    (106,  -1.25,      42.42,                42.42, 60),
    (107,  50.00, -500000.00,  5000000000000000.00, 70),
    (108, -50.00,  500000.00, -5000000000000000.00, 80),
    (109,  10.50,    1250.25,     900000000000.25, 90);

CREATE TABLE decimal_join_right_wide32 (
    rid INTEGER,
    k32_from16 DECIMAL(9,2),
    marker INTEGER
);

INSERT INTO decimal_join_right_wide32 VALUES
    (201, -99.99, 10),
    (202, -10.50, 20),
    (203,   0.00, 30),
    (204,  12.34, 40),
    (205,  99.99, 50),
    (206,  50.00, 70),
    (207, -50.00, 80);

CREATE TABLE decimal_join_empty_right (
    rid INTEGER,
    k32 DECIMAL(9,2)
);

INSERT INTO decimal_join_empty_right VALUES
    (301, 111111.11),
    (302, 222222.22);

.print '=== DECIMAL JOIN 1: DECIMAL(4,2) equality key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k16, l.payload, r.marker
     FROM decimal_join_left l
     INNER JOIN decimal_join_right r ON l.k16 = r.k16
     ORDER BY l.k16 ASC'
);

.print '=== DECIMAL JOIN 2: DECIMAL(4,2) to DECIMAL(9,2) same-scale DECIMAL32 key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k16, r.k32_from16
     FROM decimal_join_left l
     INNER JOIN decimal_join_right_wide32 r ON l.k16 = r.k32_from16
     ORDER BY l.k16 ASC'
);

.print '=== DECIMAL JOIN 3: DECIMAL(9,2) equality key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k32, l.payload, r.marker
     FROM decimal_join_left l
     INNER JOIN decimal_join_right r ON l.k32 = r.k32
     ORDER BY l.k32 ASC'
);

.print '=== DECIMAL JOIN 4: DECIMAL(18,2) equality key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k64, l.payload, r.marker
     FROM decimal_join_left l
     INNER JOIN decimal_join_right r ON l.k64 = r.k64
     ORDER BY l.k64 ASC'
);

.print '=== DECIMAL JOIN 5: DECIMAL64 key plus DECIMAL32 post-filter key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k64, l.k32 AS l_k32, r.k32 AS r_k32
     FROM decimal_join_left l
     INNER JOIN decimal_join_right r ON l.k64 = r.k64 AND l.k32 = r.k32
     ORDER BY l.id ASC'
);

.print '=== DECIMAL JOIN 6: no matching DECIMAL(9,2) keys ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid
     FROM decimal_join_left l
     INNER JOIN decimal_join_empty_right r ON l.k32 = r.k32
     ORDER BY l.id ASC'
);

-- ============================================================================
-- DECIMAL GROUP BY + aggregate tests
-- ============================================================================

CREATE TABLE decimal_groupby (
    grp32 DECIMAL(9,2),
    grp64 DECIMAL(18,2),
    val32 DECIMAL(9,2),
    val64 DECIMAL(18,2),
    marker INTEGER
);

INSERT INTO decimal_groupby VALUES
    (-10.50, -900000000000.25, -1250.25, -900000000000.00, 1),
    (-10.50, -900000000000.25,   -42.42, -600000000000.00, 2),
    (-10.50, -900000000000.25,     0.01, -300000000000.00, 3),
    (  0.00,             0.01,    -0.01,            -1.00, 4),
    (  0.00,             0.01,    42.42,             2.00, 5),
    (  0.00,             0.01,  1250.25,             5.00, 6),
    ( 12.34, 1234567890123456.78, -999999.99,  300000000000.00, 7),
    ( 12.34, 1234567890123456.78,   98765.43,  600000000000.00, 8),
    ( 12.34, 1234567890123456.78, 9999999.99,  900000000000.00, 9);

.print '=== DECIMAL GROUP BY 1: DECIMAL32 key count/min/max DECIMAL32 payload ==='
SELECT * FROM gpu_execution(
    'SELECT grp32, count(*) AS cnt, min(val32) AS min_val, max(val32) AS max_val
     FROM decimal_groupby
     GROUP BY grp32
     ORDER BY grp32 ASC'
);

.print '=== DECIMAL GROUP BY 2: DECIMAL64 key count(*) ==='
SELECT * FROM gpu_execution(
    'SELECT grp64, count(*) AS cnt
     FROM decimal_groupby
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

.print '=== DECIMAL GROUP BY 3: DECIMAL64 key count(decimal32 payload) ==='
SELECT * FROM gpu_execution(
    'SELECT grp64, count(val32) AS cnt_val
     FROM decimal_groupby
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

.print '=== DECIMAL GROUP BY 4: DECIMAL64 payload min/max ==='
SELECT * FROM gpu_execution(
    'SELECT grp32, min(val64) AS min64, max(val64) AS max64
     FROM decimal_groupby
     GROUP BY grp32
     ORDER BY grp32 ASC'
);

.print '=== DECIMAL GROUP BY 5: DECIMAL64 payload sum/avg ==='
SELECT * FROM gpu_execution(
    'SELECT grp32, sum(val64) AS sum64, avg(val64) AS avg64
     FROM decimal_groupby
     GROUP BY grp32
     ORDER BY grp32 ASC'
);

.print '=== DECIMAL exact transport/filter/order/join/groupby smoke tests completed ==='
