-- Numeric multi-key GROUP BY smoke tests for the new tuple-groupby design.
-- Run:
--   ./build/release/duckdb -unsigned < test/test_tuple_groupby_gpu.sql

LOAD './build/release/extension/rasterdb/rasterdb.duckdb_extension';

CREATE OR REPLACE TABLE tuple_gb_input AS
SELECT *
FROM (
  VALUES
    (1, 7, 10, 1, 100, 1000),
    (1, 7, 20, 2, 100, 1000),
    (1, 8, 5, 3, 100, 1001),
    (2, 7, 3, 4, 200, 1000),
    (2, 8, 4, 5, 200, 1001),
    (3, 7, 8, 6, 300, 1000),
    (3, 7, 2, 7, 300, 1000),
    (3, 8, 6, 8, 300, 1001),
    (3, 9, 1, 9, 300, 1002)
) AS t(k0, k1, v, w, k2, k3);

.print '=== GroupBy 1: on two INT keys, sum/count/min/max/avg ==='
SELECT * FROM gpu_execution(
  'SELECT k0, k1, sum(v) AS v_sum, count(*) AS n, min(v) AS v_min, max(v) AS v_max, avg(v) AS v_avg
   FROM tuple_gb_input
   GROUP BY k0, k1
   ORDER BY k0, k1'
);

.print '=== GroupBy 2: on three INT keys ==='
SELECT * FROM gpu_execution(
  'SELECT k0, k1, w, sum(v) AS v_sum
   FROM tuple_gb_input
   GROUP BY k0, k1, w
   ORDER BY k0, k1, w'
);

.print '=== GroupBy 3: on four INT keys, tuple path beyond old 3-key fallback ==='
SELECT * FROM gpu_execution(
  'SELECT k0, k1, k2, k3, sum(v) AS v_sum, count(*) AS n
   FROM tuple_gb_input
   GROUP BY k0, k1, k2, k3
   ORDER BY k0, k1, k2, k3'
);

.print '=== GroupBy 4: on mixed fixed-width keys and values ==='
CREATE OR REPLACE TABLE tuple_gb_mixed AS
SELECT *
FROM (
  VALUES
    (1::INTEGER, 10000000000::BIGINT, 1.5::REAL, 10.25::DOUBLE, 10::INTEGER, 100::BIGINT, 1.25::REAL, 10.5::DOUBLE),
    (1::INTEGER, 10000000000::BIGINT, 1.5::REAL, 10.25::DOUBLE, 20::INTEGER, 200::BIGINT, 2.75::REAL, 20.5::DOUBLE),
    (2::INTEGER, 20000000000::BIGINT, 2.5::REAL, 20.25::DOUBLE, 5::INTEGER, 50::BIGINT, 3.5::REAL, 30.5::DOUBLE),
    (2::INTEGER, 20000000000::BIGINT, 2.5::REAL, 20.25::DOUBLE, 7::INTEGER, 70::BIGINT, 4.5::REAL, 40.5::DOUBLE),
    (3::INTEGER, 30000000000::BIGINT, -1.5::REAL, -10.25::DOUBLE, 9::INTEGER, 90::BIGINT, -5.5::REAL, -50.5::DOUBLE)
) AS t(k32, k64, kf32, kf64, v32, v64, vf32, vf64);

SELECT * FROM gpu_execution(
  'SELECT k32, k64, kf32, kf64,
          sum(v32) AS sum_i32,
          sum(v64) AS sum_i64,
          min(vf32) AS min_f32,
          max(vf32) AS max_f32,
          avg(vf64) AS avg_f64,
          count(*) AS n
   FROM tuple_gb_mixed
   GROUP BY k32, k64, kf32, kf64
   ORDER BY k32, k64, kf32, kf64'
);

.print '=== GroupBy 5: on single FLOAT32 key, tuple path ==='
CREATE OR REPLACE TABLE tuple_gb_float32 AS
SELECT *
FROM (
  VALUES
    (1.5::REAL, 10::INTEGER, 1.25::REAL, 10.5::DOUBLE),
    (1.5::REAL, 20::INTEGER, 2.75::REAL, 20.5::DOUBLE),
    (-2.0::REAL, 5::INTEGER, -5.5::REAL, -50.5::DOUBLE),
    (-2.0::REAL, 7::INTEGER, -1.5::REAL, -10.5::DOUBLE),
    (3.25::REAL, 1::INTEGER, 3.0::REAL, 30.25::DOUBLE),
    (3.25::REAL, 2::INTEGER, 4.0::REAL, 40.25::DOUBLE),
    (3.25::REAL, 3::INTEGER, 5.0::REAL, 50.25::DOUBLE)
) AS t(kf32, v32, vf32, vf64);

SELECT * FROM gpu_execution(
  'SELECT kf32,
          sum(v32) AS v32_sum,
          count(*) AS n,
          min(v32) AS v32_min,
          max(v32) AS v32_max,
          avg(v32) AS v32_avg,
          sum(vf32) AS vf32_sum,
          min(vf32) AS vf32_min,
          max(vf32) AS vf32_max,
          avg(vf32) AS vf32_avg,
          sum(vf64) AS vf64_sum,
          min(vf64) AS vf64_min,
          max(vf64) AS vf64_max,
          avg(vf64) AS vf64_avg
   FROM tuple_gb_float32
   GROUP BY kf32
   ORDER BY kf32'
);

.print '=== GroupBy 6: on single FLOAT64 key, tuple path ==='
CREATE OR REPLACE TABLE tuple_gb_float64 AS
SELECT *
FROM (
  VALUES
    (10.25::DOUBLE, 10::INTEGER, 1.25::REAL, 10.5::DOUBLE),
    (10.25::DOUBLE, 20::INTEGER, 2.75::REAL, 20.5::DOUBLE),
    (-20.5::DOUBLE, 5::INTEGER, -5.5::REAL, -50.5::DOUBLE),
    (-20.5::DOUBLE, 7::INTEGER, -1.5::REAL, -10.5::DOUBLE),
    (30.125::DOUBLE, 1::INTEGER, 3.0::REAL, 30.25::DOUBLE),
    (30.125::DOUBLE, 2::INTEGER, 4.0::REAL, 40.25::DOUBLE),
    (30.125::DOUBLE, 3::INTEGER, 5.0::REAL, 50.25::DOUBLE)
) AS t(kf64, v32, vf32, vf64);

SELECT * FROM gpu_execution(
  'SELECT kf64,
          sum(v32) AS v32_sum,
          count(*) AS n,
          min(v32) AS v32_min,
          max(v32) AS v32_max,
          avg(v32) AS v32_avg,
          sum(vf32) AS vf32_sum,
          min(vf32) AS vf32_min,
          max(vf32) AS vf32_max,
          avg(vf32) AS vf32_avg,
          sum(vf64) AS vf64_sum,
          min(vf64) AS vf64_min,
          max(vf64) AS vf64_max,
          avg(vf64) AS vf64_avg
   FROM tuple_gb_float64
   GROUP BY kf64
   ORDER BY kf64'
);

.print '=== GroupBy 7: FLOAT +0.0/-0.0 canonicalization ==='
CREATE OR REPLACE TABLE tuple_gb_float_zero AS
SELECT *
FROM (
  VALUES
    (-0.0::REAL, -0.0::DOUBLE, 10::INTEGER),
    ( 0.0::REAL,  0.0::DOUBLE, 20::INTEGER),
    (-0.0::REAL, -0.0::DOUBLE, 30::INTEGER),
    ( 1.0::REAL,  1.0::DOUBLE, 4::INTEGER),
    ( 1.0::REAL,  1.0::DOUBLE, 6::INTEGER)
) AS t(kf32, kf64, v);

SELECT * FROM gpu_execution(
  'SELECT kf32, sum(v) AS v_sum, count(*) AS n, min(v) AS v_min, max(v) AS v_max, avg(v) AS v_avg
   FROM tuple_gb_float_zero
   GROUP BY kf32
   ORDER BY kf32'
);

SELECT * FROM gpu_execution(
  'SELECT kf64, sum(v) AS v_sum, count(*) AS n, min(v) AS v_min, max(v) AS v_max, avg(v) AS v_avg
   FROM tuple_gb_float_zero
   GROUP BY kf64
   ORDER BY kf64'
);

.print '=== GroupBy 8: on mixed INT + FLOAT keys with repeated values ==='
CREATE OR REPLACE TABLE tuple_gb_int_float AS
SELECT *
FROM (
  VALUES
    (1::INTEGER, 0.5::REAL, 10.0::DOUBLE, 3::INTEGER),
    (1::INTEGER, 0.5::REAL, 10.0::DOUBLE, 7::INTEGER),
    (1::INTEGER, 1.5::REAL, 20.0::DOUBLE, 11::INTEGER),
    (2::INTEGER, -0.0::REAL, -0.0::DOUBLE, 5::INTEGER),
    (2::INTEGER, 0.0::REAL, 0.0::DOUBLE, 15::INTEGER),
    (2::INTEGER, 2.5::REAL, -30.5::DOUBLE, 2::INTEGER),
    (2::INTEGER, 2.5::REAL, -30.5::DOUBLE, 8::INTEGER)
) AS t(k32, kf32, kf64, v);

SELECT * FROM gpu_execution(
  'SELECT k32, kf32, kf64, sum(v) AS v_sum, count(*) AS n, min(v) AS v_min, max(v) AS v_max, avg(v) AS v_avg
   FROM tuple_gb_int_float
   GROUP BY k32, kf32, kf64
   ORDER BY k32, kf32, kf64'
);

.print '=== GroupBy tests completed ==='
