-- RasterDB TPC-H adapted query suite.
-- First run:
--   python3 create_tpch_adapted_db.py --sf 25
-- Run with:
--   rduckdb tpch_sf25.db < tpch_adapted.sql

SET enable_duckdb_fallback = true;

-- Q1: Q1_pricing_summary_int
-- Features: GROUP BY + aggregate + filter
-- Adaptation: VARCHAR group keys replaced with integer IDs; DATE filter replaced with YYYYMMDD integer; avg and discount/charge aggregates removed.
SELECT * FROM gpu_execution('
SELECT
  l_returnflag_id,
  l_linestatus_id,
  sum(l_quantity) AS sum_qty,
  sum(l_extendedprice) AS sum_base_price,
  count(*) AS count_order
FROM lineitem_int
WHERE l_shipdate_int <= 19980902
GROUP BY l_returnflag_id, l_linestatus_id
');

-- Q3: Q3_shipping_priority_int
-- Features: 3-way JOIN + filter + GROUP BY + ORDER BY
-- Adaptation: VARCHAR market segment filter replaced with c_mktsegment_id; DATE filters replaced with YYYYMMDD integers; LIMIT removed.
SELECT * FROM gpu_execution('
SELECT
  l.l_orderkey,
  sum(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue,
  o.o_orderdate_int,
  o.o_shippriority
FROM lineitem_int l
INNER JOIN orders_int o ON l.l_orderkey = o.o_orderkey
INNER JOIN customer_int c ON c.c_custkey = o.o_custkey
WHERE c.c_mktsegment_id = 2
  AND o.o_orderdate_int < 19950315
  AND l.l_shipdate_int > 19950315
GROUP BY l.l_orderkey, o.o_orderdate_int, o.o_shippriority
ORDER BY revenue DESC, o.o_orderdate_int
');

-- Q4: Q4_order_priority_int
-- Features: semi-join rewrite + DATE filter + GROUP BY + COUNT + ORDER BY
-- Adaptation: order priority encoded as o_orderpriority_id; EXISTS rewritten as grouped lineitem keys joined to orders; DATE filters replaced with YYYYMMDD integers.
SELECT * FROM gpu_execution('
SELECT
  o.o_orderpriority_id,
  count(*) AS order_count
FROM orders_int o
INNER JOIN (
  SELECT l_orderkey
  FROM lineitem_int
  WHERE l_commitdate_int < l_receiptdate_int
  GROUP BY l_orderkey
) li ON o.o_orderkey = li.l_orderkey
WHERE o.o_orderdate_int >= 19930701
  AND o.o_orderdate_int < 19931001
GROUP BY o.o_orderpriority_id
ORDER BY o.o_orderpriority_id
');

-- Q5: Q5_local_supplier_volume_int
-- Features: 6-way JOIN + filter + GROUP BY + ORDER BY
-- Adaptation: nation and region names replaced with integer IDs; DATE filters replaced with YYYYMMDD integers.
SELECT * FROM gpu_execution('
SELECT
  n.n_name_id,
  sum(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue
FROM lineitem_int l
INNER JOIN orders_int o ON l.l_orderkey = o.o_orderkey
INNER JOIN customer_int c ON c.c_custkey = o.o_custkey
INNER JOIN supplier_int s
  ON l.l_suppkey = s.s_suppkey
 AND c.c_nationkey = s.s_nationkey
INNER JOIN nation_int n ON s.s_nationkey = n.n_nationkey
INNER JOIN region_int r ON n.n_regionkey = r.r_regionkey
WHERE r.r_name_id = 2
  AND o.o_orderdate_int >= 19940101
  AND o.o_orderdate_int < 19950101
GROUP BY n.n_name_id
ORDER BY revenue DESC
');

-- Q6: Q6_forecasting_revenue_int
-- Features: filter + SUM
-- Adaptation: DATE filter replaced with YYYYMMDD integer; BETWEEN rewritten as explicit comparisons.
SELECT * FROM gpu_execution('
SELECT
  sum(l_extendedprice * l_discount) AS revenue
FROM lineitem_int
WHERE l_shipdate_int >= 19940101
  AND l_shipdate_int < 19950101
  AND l_discount > 0.05
  AND l_discount < 0.07
  AND l_quantity < 24.0
');

-- Q7: Q7_volume_shipping_int
-- Features: 6-way JOIN + numeric pair filter + projected year + GROUP BY + SUM + ORDER BY
-- Adaptation: FRANCE/GERMANY replaced with n_name_id values 6/7; EXTRACT(YEAR) replaced with integer projection; DATE range replaced with YYYYMMDD integers.
SELECT * FROM gpu_execution('
SELECT
  supp_nation,
  cust_nation,
  l_year,
  sum(volume) AS revenue
FROM (
  SELECT
    n1.n_name_id AS supp_nation,
    n2.n_name_id AS cust_nation,
    CAST(l.l_shipdate_int / 10000 AS INTEGER) AS l_year,
    l.l_extendedprice * (1.0 - l.l_discount) AS volume
  FROM supplier_int s
  INNER JOIN lineitem_int l ON s.s_suppkey = l.l_suppkey
  INNER JOIN orders_int o ON o.o_orderkey = l.l_orderkey
  INNER JOIN customer_int c ON c.c_custkey = o.o_custkey
  INNER JOIN nation_int n1 ON s.s_nationkey = n1.n_nationkey
  INNER JOIN nation_int n2 ON c.c_nationkey = n2.n_nationkey
  WHERE n1.n_name_id >= 6
    AND n1.n_name_id <= 7
    AND n2.n_name_id >= 6
    AND n2.n_name_id <= 7
    AND n1.n_name_id <> n2.n_name_id
    AND l.l_shipdate_int >= 19950101
    AND l.l_shipdate_int <= 19961231
) shipping
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year
');

-- Q10: Q10_returned_item_int
-- Features: 4-way JOIN + filter + GROUP BY + SUM + ORDER BY
-- Adaptation: VARCHAR output columns removed; return flag replaced with l_returnflag_id; DATE filter replaced with YYYYMMDD integer; LIMIT removed.
SELECT * FROM gpu_execution('
SELECT
  c.c_custkey,
  sum(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue,
  c.c_acctbal,
  n.n_nationkey
FROM customer_int c
INNER JOIN orders_int o ON c.c_custkey = o.o_custkey
INNER JOIN lineitem_int l ON l.l_orderkey = o.o_orderkey
INNER JOIN nation_int n ON c.c_nationkey = n.n_nationkey
WHERE o.o_orderdate_int >= 19931001
  AND o.o_orderdate_int < 19940101
  AND l.l_returnflag_id = 3
GROUP BY c.c_custkey, c.c_acctbal, n.n_nationkey
ORDER BY revenue DESC
');

-- Q11: Q11_important_stock_int
-- Features: 3-way JOIN + filter + GROUP BY + SUM + ORDER BY
-- Adaptation: nation name filter replaced with n_name_id; HAVING subquery removed.
SELECT * FROM gpu_execution('
SELECT
  ps.ps_partkey,
  sum(ps.ps_supplycost * ps.ps_availqty) AS value
FROM partsupp_int ps
INNER JOIN supplier_int s ON ps.ps_suppkey = s.s_suppkey
INNER JOIN nation_int n ON s.s_nationkey = n.n_nationkey
WHERE n.n_name_id = 7
GROUP BY ps.ps_partkey
ORDER BY value DESC
');

-- Q12: Q12_shipping_modes_int
-- Features: 2-way JOIN + filter + GROUP BY + COUNT
-- Adaptation: ship mode IN filter replaced with integer range; CASE aggregates simplified to count; DATE filters replaced with YYYYMMDD integers.
SELECT * FROM gpu_execution('
SELECT
  l.l_shipmode_id,
  count(*) AS cnt
FROM orders_int o
INNER JOIN lineitem_int l ON o.o_orderkey = l.l_orderkey
WHERE l.l_shipmode_id >= 4
  AND l.l_shipmode_id <= 6
  AND l.l_commitdate_int < l.l_receiptdate_int
  AND l.l_shipdate_int < l.l_commitdate_int
  AND l.l_receiptdate_int >= 19940101
  AND l.l_receiptdate_int < 19950101
GROUP BY l.l_shipmode_id
');

-- Q14: Q14_promotion_effect_int
-- Features: 2-way JOIN + filter + ungrouped SUM
-- Adaptation: promo CASE/LIKE split removed; simplified to total revenue; DATE filter replaced with YYYYMMDD integer.
SELECT * FROM gpu_execution('
SELECT
  sum(l.l_extendedprice * (1.0 - l.l_discount)) AS total_revenue
FROM lineitem_int l
INNER JOIN part_int p ON l.l_partkey = p.p_partkey
WHERE l.l_shipdate_int >= 19950901
  AND l.l_shipdate_int < 19951001
');

-- Q18: Q18_large_volume_customer_int
-- Features: 3-way JOIN + GROUP BY + HAVING + ORDER BY + LIMIT
-- Adaptation: customer name removed; date output replaced with o_orderdate_int; IN/HAVING subquery expressed as grouped lineitem subquery joined to orders and customer.
SELECT * FROM gpu_execution('
SELECT
  c.c_custkey,
  o.o_orderkey,
  o.o_orderdate_int,
  o.o_totalprice,
  oq.sum_quantity
FROM (
  SELECT
    l_orderkey,
    sum(l_quantity) AS sum_quantity
  FROM lineitem_int
  GROUP BY l_orderkey
  HAVING sum(l_quantity) > 300.0
) oq
INNER JOIN orders_int o ON oq.l_orderkey = o.o_orderkey
INNER JOIN customer_int c ON o.o_custkey = c.c_custkey
ORDER BY o.o_totalprice DESC, o.o_orderdate_int
LIMIT 100
');
