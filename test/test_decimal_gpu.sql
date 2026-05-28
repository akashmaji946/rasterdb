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

.print '=== DECIMAL exact transport/filter smoke tests completed ==='
