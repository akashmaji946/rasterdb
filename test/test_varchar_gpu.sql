-- Test VARCHAR/STRING GPU operations end-to-end
-- Run with: ./build/release/duckdb -unsigned < test/test_varchar_gpu.sql

-- Load extension
LOAD '/home/akashmaji/Device/IMPORTANT/rasterdb/build/release/extension/rasterdb/rasterdb.duckdb_extension';

-- Generate TPC-H SF0.01 data using built-in tpch extension
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=0.01);

-- Init GPU buffers (skip if no discrete GPU available)
-- CALL gpu_buffer_init('1 GB', '1 GB');

-- ============================================================================
-- Test 1: Simple VARCHAR filter (nation table — 25 rows, all VARCHAR)
-- ============================================================================
.print '=== Test 1: VARCHAR filter on nation ==='
SELECT * FROM gpu_execution('SELECT n_nationkey, n_name FROM nation WHERE n_name = ''BRAZIL''');

-- ============================================================================
-- Test 2: VARCHAR filter on region (5 rows)
-- ============================================================================
.print '=== Test 2: VARCHAR filter on region ==='
SELECT * FROM gpu_execution('SELECT r_regionkey, r_name FROM region WHERE r_name = ''EUROPE''');

-- ============================================================================
-- Test 3: GROUP BY on VARCHAR key (nation names)
-- ============================================================================
.print '=== Test 3: GROUP BY VARCHAR key ==='
SELECT * FROM gpu_execution('SELECT n_name, count(*) as cnt FROM nation GROUP BY n_name');

-- ============================================================================
-- Test 4: JOIN with VARCHAR keys (nation ⋈ region via integer keys, but with
-- VARCHAR columns in the output)
-- ============================================================================
.print '=== Test 4: JOIN with VARCHAR in output ==='
SELECT * FROM gpu_execution('SELECT n.n_name, r.r_name FROM nation n INNER JOIN region r ON n.n_regionkey = r.r_regionkey LIMIT 10');

-- ============================================================================
-- Test 5: Filter + GROUP BY on customer table (has c_mktsegment VARCHAR)
-- ============================================================================
.print '=== Test 5: Customer mktsegment groupby ==='
SELECT * FROM gpu_execution('SELECT c_mktsegment, count(*) as cnt FROM customer GROUP BY c_mktsegment');

-- ============================================================================
-- Test 6: TPC-H Q5 simplified — join with VARCHAR filter on region
-- ============================================================================
.print '=== Test 6: Simplified Q5 - join nation+region with VARCHAR filter ==='
SELECT * FROM gpu_execution('SELECT n.n_name, n.n_nationkey FROM nation n INNER JOIN region r ON n.n_regionkey = r.r_regionkey WHERE r.r_name = ''EUROPE''');

.print '=== All VARCHAR GPU tests completed ==='
