-- Test exact DECIMAL GPU support by physical decimal width.
-- Run with:
--   ./build/release/duckdb -unsigned < test/test_decimal_gpu.sql
--
-- Layout:
--   D16  = DECIMAL(4,2), DuckDB INT16 storage widened to RasterDF DECIMAL32.
--   D32  = DECIMAL(9,2), RasterDF INT32 fixed-point storage.
--   D64  = DECIMAL(18,2), RasterDF INT64 fixed-point storage.
--   D128 = DECIMAL(38,4), RasterDF INT128 fixed-point storage.

LOAD '~/Device/IMPORTANT/rasterdb/build/release/extension/rasterdb/rasterdb.duckdb_extension';

-- ============================================================================
-- D16: DECIMAL(4,2)
-- ============================================================================

CREATE TABLE decimal16_exact (
    id INTEGER,
    d16 DECIMAL(4,2)
);

INSERT INTO decimal16_exact VALUES
    (1,  -0.05),
    (2,   0.00),
    (3,  12.34),
    (4, -99.99),
    (5,  99.99),
    (6, -10.50),
    (7,  10.50),
    (8,   0.01),
    (9,  -1.25),
    (10,  1.25),
    (11,-50.00),
    (12, 50.00);

.print '=== D16 projection/readback: DECIMAL(4,2), INT16 widened to DECIMAL32 ==='
SELECT * FROM gpu_execution(
    'SELECT id, d16 FROM decimal16_exact ORDER BY id'
);

.print '=== D16 filter: scaled narrow values ==='
SELECT * FROM gpu_execution(
    'SELECT id, d16 FROM decimal16_exact
     WHERE d16 >= -0.05 AND d16 <= 12.34
     ORDER BY id'
);

.print '=== D16 ORDER BY ASC: signed DECIMAL32 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d16 FROM decimal16_exact ORDER BY d16 ASC'
);

CREATE TABLE decimal16_join_left (
    id INTEGER,
    k16 DECIMAL(4,2),
    payload INTEGER
);

CREATE TABLE decimal16_join_right (
    rid INTEGER,
    k16 DECIMAL(4,2),
    marker INTEGER
);

INSERT INTO decimal16_join_left VALUES
    (1, -99.99, 100),
    (2, -10.50, 200),
    (3,   0.00, 300),
    (4,  12.34, 400),
    (5,  99.99, 500),
    (6,   1.25, 600),
    (7,  50.00, 700),
    (8, -50.00, 800);

INSERT INTO decimal16_join_right VALUES
    (101, -99.99, 10),
    (102, -10.50, 20),
    (103,   0.00, 30),
    (104,  12.34, 40),
    (105,  99.99, 50),
    (106,  -1.25, 60),
    (107,  50.00, 70),
    (108, -50.00, 80),
    (109,  10.50, 90);

.print '=== D16 JOIN: DECIMAL(4,2) equality key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k16, l.payload, r.marker
     FROM decimal16_join_left l
     INNER JOIN decimal16_join_right r ON l.k16 = r.k16
     ORDER BY l.k16 ASC'
);

-- ============================================================================
-- D32: DECIMAL(9,2)
-- ============================================================================

CREATE TABLE decimal32_exact (
    id INTEGER,
    d32 DECIMAL(9,2)
);

INSERT INTO decimal32_exact VALUES
    (1,   -1250.25),
    (2,       0.01),
    (3,   98765.43),
    (4, -999999.99),
    (5, 9999999.99),
    (6,      -0.01),
    (7,    1250.25),
    (8,  -98765.43),
    (9,      42.42),
    (10,    -42.42),
    (11, 500000.00),
    (12,-500000.00);

.print '=== D32 projection/readback: DECIMAL(9,2), INT32 fixed-point ==='
SELECT * FROM gpu_execution(
    'SELECT id, d32 FROM decimal32_exact ORDER BY id'
);

.print '=== D32 filter: scaled INT32 values ==='
SELECT * FROM gpu_execution(
    'SELECT id, d32 FROM decimal32_exact
     WHERE d32 >= -1250.25 AND d32 <= 1250.25
     ORDER BY d32 ASC'
);

.print '=== D32 ORDER BY DESC: signed DECIMAL32 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d32 FROM decimal32_exact ORDER BY d32 DESC'
);

.print '=== D32 ORDER BY ASC: signed DECIMAL32 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d32 FROM decimal32_exact ORDER BY d32 ASC'
);

CREATE TABLE decimal32_join_left (
    id INTEGER,
    k32 DECIMAL(9,2),
    payload INTEGER
);

CREATE TABLE decimal32_join_right (
    rid INTEGER,
    k32 DECIMAL(9,2),
    marker INTEGER
);

INSERT INTO decimal32_join_left VALUES
    (1, -999999.99, 100),
    (2,   -1250.25, 200),
    (3,       0.01, 300),
    (4,   98765.43, 400),
    (5, 9999999.99, 500),
    (6,     -42.42, 600),
    (7, -500000.00, 700),
    (8,  500000.00, 800);

INSERT INTO decimal32_join_right VALUES
    (101, -999999.99, 10),
    (102,   -1250.25, 20),
    (103,       0.01, 30),
    (104,   98765.43, 40),
    (105, 9999999.99, 50),
    (106,      42.42, 60),
    (107, -500000.00, 70),
    (108,  500000.00, 80),
    (109,    1250.25, 90);

CREATE TABLE decimal32_join_empty_right (
    rid INTEGER,
    k32 DECIMAL(9,2)
);

INSERT INTO decimal32_join_empty_right VALUES
    (301, 111111.11),
    (302, 222222.22);

.print '=== D32 JOIN: DECIMAL(9,2) equality key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k32, l.payload, r.marker
     FROM decimal32_join_left l
     INNER JOIN decimal32_join_right r ON l.k32 = r.k32
     ORDER BY l.k32 ASC'
);

.print '=== D32 JOIN: no matching DECIMAL(9,2) keys ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid
     FROM decimal32_join_left l
     INNER JOIN decimal32_join_empty_right r ON l.k32 = r.k32
     ORDER BY l.id ASC'
);

CREATE TABLE decimal32_groupby (
    grp32 DECIMAL(9,2),
    val32 DECIMAL(9,2),
    marker INTEGER
);

INSERT INTO decimal32_groupby VALUES
    (-10.50,   -1250.25, 1),
    (-10.50,     -42.42, 2),
    (-10.50,       0.01, 3),
    (  0.00,      -0.01, 4),
    (  0.00,      42.42, 5),
    (  0.00,    1250.25, 6),
    ( 12.34, -999999.99, 7),
    ( 12.34,   98765.43, 8),
    ( 12.34, 9999999.99, 9);

.print '=== D32 GROUP BY: DECIMAL32 key count/min/max DECIMAL32 payload ==='
SELECT * FROM gpu_execution(
    'SELECT grp32, count(*) AS cnt, min(val32) AS min_val, max(val32) AS max_val
     FROM decimal32_groupby
     GROUP BY grp32
     ORDER BY grp32 ASC'
);

-- ============================================================================
-- D16 <-> D32 same-scale compatibility
-- ============================================================================

CREATE TABLE decimal16_to_decimal32_right (
    rid INTEGER,
    k32_from16 DECIMAL(9,2),
    marker INTEGER
);

INSERT INTO decimal16_to_decimal32_right VALUES
    (201, -99.99, 10),
    (202, -10.50, 20),
    (203,   0.00, 30),
    (204,  12.34, 40),
    (205,  99.99, 50),
    (206,  50.00, 70),
    (207, -50.00, 80);

.print '=== D16/D32 JOIN: DECIMAL(4,2) to DECIMAL(9,2) same-scale key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k16, r.k32_from16
     FROM decimal16_join_left l
     INNER JOIN decimal16_to_decimal32_right r ON l.k16 = r.k32_from16
     ORDER BY l.k16 ASC'
);

-- ============================================================================
-- D64: DECIMAL(18,2)
-- ============================================================================

CREATE TABLE decimal64_exact (
    id INTEGER,
    d64 DECIMAL(18,2)
);

INSERT INTO decimal64_exact VALUES
    (1,     -900000000000.25),
    (2,                 0.01),
    (3,  1234567890123456.78),
    (4, -9999999999999999.99),
    (5,  9999999999999999.99),
    (6,                -0.01),
    (7,      900000000000.25),
    (8, -1234567890123456.78),
    (9,                42.42),
    (10,              -42.42),
    (11,-5000000000000000.00),
    (12, 5000000000000000.00);

.print '=== D64 projection/readback: DECIMAL(18,2), INT64 fixed-point ==='
SELECT * FROM gpu_execution(
    'SELECT id, d64 FROM decimal64_exact ORDER BY id'
);

.print '=== D64 filter: scaled INT64 values ==='
SELECT * FROM gpu_execution(
    'SELECT id, d64 FROM decimal64_exact
     WHERE d64 >= -900000000000.25 AND d64 <= 900000000000.25
     ORDER BY d64 ASC'
);

.print '=== D64 ORDER BY ASC: signed DECIMAL64 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d64 FROM decimal64_exact ORDER BY d64 ASC'
);

.print '=== D64 ORDER BY DESC: signed DECIMAL64 keys ==='
SELECT * FROM gpu_execution(
    'SELECT id, d64 FROM decimal64_exact ORDER BY d64 DESC'
);

CREATE TABLE decimal64_join_left (
    id INTEGER,
    k64 DECIMAL(18,2),
    payload INTEGER
);

CREATE TABLE decimal64_join_right (
    rid INTEGER,
    k64 DECIMAL(18,2),
    marker INTEGER
);

INSERT INTO decimal64_join_left VALUES
    (1, -9999999999999999.99, 100),
    (2,    -900000000000.25, 200),
    (3,                 0.01, 300),
    (4,  1234567890123456.78, 400),
    (5,  9999999999999999.99, 500),
    (6,               -42.42, 600),
    (7,  5000000000000000.00, 700),
    (8, -5000000000000000.00, 800);

INSERT INTO decimal64_join_right VALUES
    (101, -9999999999999999.99, 10),
    (102,    -900000000000.25, 20),
    (103,                 0.01, 30),
    (104,  1234567890123456.78, 40),
    (105,  9999999999999999.99, 50),
    (106,                42.42, 60),
    (107,  5000000000000000.00, 70),
    (108, -5000000000000000.00, 80),
    (109,      900000000000.25, 90);

.print '=== D64 JOIN: DECIMAL(18,2) equality key ==='
SELECT * FROM gpu_execution(
    'SELECT l.id, r.rid, l.k64, l.payload, r.marker
     FROM decimal64_join_left l
     INNER JOIN decimal64_join_right r ON l.k64 = r.k64
     ORDER BY l.k64 ASC'
);

CREATE TABLE decimal64_groupby (
    grp64 DECIMAL(18,2),
    val64 DECIMAL(18,2),
    marker INTEGER
);

INSERT INTO decimal64_groupby VALUES
    (-900000000000.25, -900000000000.00, 1),
    (-900000000000.25, -600000000000.00, 2),
    (-900000000000.25, -300000000000.00, 3),
    (            0.01,            -1.00, 4),
    (            0.01,             2.00, 5),
    (            0.01,             5.00, 6),
    (1234567890123456.78, 300000000000.00, 7),
    (1234567890123456.78, 600000000000.00, 8),
    (1234567890123456.78, 900000000000.00, 9);

.print '=== D64 GROUP BY: DECIMAL64 key count(*) ==='
SELECT * FROM gpu_execution(
    'SELECT grp64, count(*) AS cnt
     FROM decimal64_groupby
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

.print '=== D64 GROUP BY: DECIMAL64 payload min/max ==='
SELECT * FROM gpu_execution(
    'SELECT grp64, min(val64) AS min64, max(val64) AS max64
     FROM decimal64_groupby
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

.print '=== D64 GROUP BY: DECIMAL64 payload sum/avg ==='
SELECT * FROM gpu_execution(
    'SELECT grp64, sum(val64) AS sum64, avg(val64) AS avg64
     FROM decimal64_groupby
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

-- ============================================================================
-- Cross-width decimal GROUP BY matrix
-- ============================================================================

CREATE TABLE decimal_groupby_cross (
    id INTEGER,
    grp16 DECIMAL(4,2),
    grp32 DECIMAL(9,2),
    grp64 DECIMAL(18,2),
    val16 DECIMAL(4,2),
    val32 DECIMAL(9,2),
    val64 DECIMAL(18,2),
    marker INTEGER
);

INSERT INTO decimal_groupby_cross VALUES
    (1, -10.50,   -1250.25,    -900000000000.25,  -1.25,   -1250.25, -900000000000.00, 10),
    (2, -10.50,   -1250.25,    -900000000000.25,   2.50,     -42.42, -600000000000.00, 20),
    (3, -10.50,   -1250.25,    -900000000000.25,   5.00,       0.01, -300000000000.00, 30),
    (4,   0.00,       0.01,                 0.01, -10.00,      -0.01,            -1.00, 40),
    (5,   0.00,       0.01,                 0.01,   0.00,      42.42,             2.00, 50),
    (6,   0.00,       0.01,                 0.01,  10.00,    1250.25,             5.00, 60),
    (7,  12.34,   98765.43,  1234567890123456.78, -50.00, -999999.99,  300000000000.00, 70),
    (8,  12.34,   98765.43,  1234567890123456.78,  25.00,   98765.43,  600000000000.00, 80),
    (9,  12.34,   98765.43,  1234567890123456.78,  50.00, 9999999.99,  900000000000.00, 90);

.print '=== CROSS GROUP BY 1: D16 key with D32 payload sum/min/max/avg ==='
SELECT * FROM gpu_execution(
    'SELECT grp16,
            count(*) AS cnt,
            sum(val32) AS sum32,
            min(val32) AS min32,
            max(val32) AS max32,
            avg(val32) AS avg32
     FROM decimal_groupby_cross
     GROUP BY grp16
     ORDER BY grp16 ASC'
);

.print '=== CROSS GROUP BY 2: D16 key with D64 payload sum/min/max/avg ==='
SELECT * FROM gpu_execution(
    'SELECT grp16,
            sum(val64) AS sum64,
            min(val64) AS min64,
            max(val64) AS max64,
            avg(val64) AS avg64
     FROM decimal_groupby_cross
     GROUP BY grp16
     ORDER BY grp16 ASC'
);

.print '=== CROSS GROUP BY 3: D32 key with D16 payload count/sum/min/max/avg ==='
SELECT * FROM gpu_execution(
    'SELECT grp32,
            count(*) AS cnt,
            sum(val16) AS sum16,
            min(val16) AS min16,
            max(val16) AS max16,
            avg(val16) AS avg16
     FROM decimal_groupby_cross
     GROUP BY grp32
     ORDER BY grp32 ASC'
);

.print '=== CROSS GROUP BY 4: D32 key with D64 payload sum/min/max/avg ==='
SELECT * FROM gpu_execution(
    'SELECT grp32,
            sum(val64) AS sum64,
            min(val64) AS min64,
            max(val64) AS max64,
            avg(val64) AS avg64
     FROM decimal_groupby_cross
     GROUP BY grp32
     ORDER BY grp32 ASC'
);

.print '=== CROSS GROUP BY 5: D64 key with D16 payload count/sum/min/max/avg ==='
SELECT * FROM gpu_execution(
    'SELECT grp64,
            count(*) AS cnt,
            sum(val16) AS sum16,
            min(val16) AS min16,
            max(val16) AS max16,
            avg(val16) AS avg16
     FROM decimal_groupby_cross
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

.print '=== CROSS GROUP BY 6: D64 key with D32 payload count/sum/min/max/avg ==='
SELECT * FROM gpu_execution(
    'SELECT grp64,
            count(*) AS cnt,
            sum(val32) AS sum32,
            min(val32) AS min32,
            max(val32) AS max32,
            avg(val32) AS avg32
     FROM decimal_groupby_cross
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

.print '=== CROSS GROUP BY 7: D64 key with mixed D16/D32/D64 aggregate payloads ==='
SELECT * FROM gpu_execution(
    'SELECT grp64,
            min(val16) AS min16,
            max(val32) AS max32,
            sum(val64) AS sum64,
            count(*) AS cnt
     FROM decimal_groupby_cross
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

CREATE TABLE decimal_groupby_cross_sparse (
    grp64 DECIMAL(18,2),
    val32 DECIMAL(9,2),
    val64 DECIMAL(18,2)
);

INSERT INTO decimal_groupby_cross_sparse VALUES
    (-5000000000000000.00, -999999.99, -100000000000.00),
    (               -0.01,      -0.01,             -0.01),
    (                0.01,       0.01,              0.01),
    ( 5000000000000000.00,  999999.99,  100000000000.00);

.print '=== CROSS GROUP BY 8: D64 key sparse one-row groups with D32/D64 payloads ==='
SELECT * FROM gpu_execution(
    'SELECT grp64,
            count(*) AS cnt,
            sum(val32) AS sum32,
            min(val64) AS min64,
            max(val64) AS max64
     FROM decimal_groupby_cross_sparse
     GROUP BY grp64
     ORDER BY grp64 ASC'
);

-- -- ============================================================================
-- -- D128: DECIMAL(38,4)
-- -- ============================================================================

-- CREATE TABLE decimal128_exact (
--     id INTEGER,
--     d128 DECIMAL(38,4)
-- );

-- INSERT INTO decimal128_exact VALUES
--     (1, -9999999999999999999999999999999999.9999),
--     (2, -123456789012345678901234567890.1234),
--     (3, 0.0001),
--     (4, 123456789012345678901234567890.1234),
--     (5, 9999999999999999999999999999999999.9999);

-- .print '=== D128 projection/readback: DECIMAL(38,4), exact 16-byte fixed-point ==='
-- SELECT * FROM gpu_execution(
--     'SELECT id, d128 FROM decimal128_exact ORDER BY id'
-- );

-- .print '=== D128 filter: signed huge fixed-point range ==='
-- SELECT * FROM gpu_execution(
--     'SELECT id, d128
--      FROM decimal128_exact
--      WHERE d128 >= -123456789012345678901234567890.1234
--        AND d128 <= 123456789012345678901234567890.1234
--      ORDER BY id'
-- );

-- .print '=== D128 filter: positive high-range values ==='
-- SELECT * FROM gpu_execution(
--     'SELECT id, d128
--      FROM decimal128_exact
--      WHERE d128 > 100000000000000000000000000000.0000
--      ORDER BY id'
-- );

-- .print '=== D128 ORDER BY ASC: signed DECIMAL128 keys ==='
-- SELECT * FROM gpu_execution(
--     'SELECT id, d128 FROM decimal128_exact ORDER BY d128 ASC'
-- );

-- .print '=== D128 ORDER BY DESC: signed DECIMAL128 keys ==='
-- SELECT * FROM gpu_execution(
--     'SELECT id, d128 FROM decimal128_exact ORDER BY d128 DESC'
-- );

-- CREATE TABLE decimal128_join_left (
--     id INTEGER,
--     k128 DECIMAL(38,4),
--     payload INTEGER
-- );

-- CREATE TABLE decimal128_join_right (
--     rid INTEGER,
--     k128 DECIMAL(38,4),
--     marker INTEGER
-- );

-- INSERT INTO decimal128_join_left VALUES
--     (1, -9999999999999999999999999999999999.9999, 100),
--     (2, -123456789012345678901234567890.1234, 200),
--     (3,                                  0.0001, 300),
--     (4,  123456789012345678901234567890.1234, 400),
--     (5,  9999999999999999999999999999999999.9999, 500),
--     (6,                                -42.4200, 600),
--     (7,                                 42.4200, 700);

-- INSERT INTO decimal128_join_right VALUES
--     (101, -9999999999999999999999999999999999.9999, 10),
--     (102, -123456789012345678901234567890.1234, 20),
--     (103,                                  0.0001, 30),
--     (104,  123456789012345678901234567890.1234, 40),
--     (105,  9999999999999999999999999999999999.9999, 50),
--     (106,                                -42.4200, 60),
--     (107,                                 99.9900, 70);

-- .print '=== D128 JOIN: DECIMAL(38,4) equality key ==='
-- SELECT * FROM gpu_execution(
--     'SELECT l.id, r.rid, l.k128, l.payload, r.marker
--      FROM decimal128_join_left l
--      INNER JOIN decimal128_join_right r ON l.k128 = r.k128
--      ORDER BY l.k128 ASC'
-- );

-- CREATE TABLE decimal128_groupby (
--     grp128 DECIMAL(38,4),
--     val128 DECIMAL(38,4),
--     marker INTEGER
-- );

-- INSERT INTO decimal128_groupby VALUES
--     (-123456789012345678901234567890.1234, -100000000000000000000.0000, 1),
--     (-123456789012345678901234567890.1234,                    -42.4200, 2),
--     (-123456789012345678901234567890.1234,                      0.0001, 3),
--     (                                  0.0001,                  -1.2500, 4),
--     (                                  0.0001,                   2.5000, 5),
--     (                                  0.0001,                   5.0000, 6),
--     ( 123456789012345678901234567890.1234,                      7.7500, 7),
--     ( 123456789012345678901234567890.1234,                     10.2500, 8),
--     ( 123456789012345678901234567890.1234,  100000000000000000000.0000, 9);

-- .print '=== D128 GROUP BY: DECIMAL128 key count/sum/min/max DECIMAL128 payload ==='
-- SELECT * FROM gpu_execution(
--     'SELECT grp128,
--             count(*) AS cnt,
--             sum(val128) AS sum128,
--             min(val128) AS min128,
--             max(val128) AS max128
--      FROM decimal128_groupby
--      GROUP BY grp128
--      ORDER BY grp128 ASC'
-- );

-- .print '=== DECIMAL exact transport/filter/order/join/groupby smoke tests completed ==='
