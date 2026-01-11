-- TPC-H Query 5 - GPU Caching and Processing

-- Cache columns for each table
call gpu_caching("orders", ["o_orderkey", "o_custkey", "o_orderdate"]);
call gpu_caching("lineitem", ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]);
call gpu_caching("supplier", ["s_suppkey", "s_nationkey"]);
call gpu_caching("nation", ["n_nationkey", "n_name", "n_regionkey"]);
call gpu_caching("region", ["r_regionkey", "r_name"]);
call gpu_caching("customer", ["c_custkey", "c_nationkey"]);

-- Execute query on GPU
call gpu_processing("select
  n_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  customer,
  orders,
  lineitem,
  supplier,
  nation,
  region
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'ASIA'
  and o_orderdate >= 19940101
  and o_orderdate <= 19941231
group by
  n_name
order by
  revenue desc;");
