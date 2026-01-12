-- TPC-H Query 10 - GPU Caching and Processing
call gpu_buffer_init('180 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("lineitem", ["l_orderkey", "l_extendedprice", "l_discount", "l_returnflag"]);
call gpu_caching("orders", ["o_orderkey", "o_custkey", "o_orderdate"]);
call gpu_caching("nation", ["n_nationkey", "n_name"]);
call gpu_processing_resize('85 GB', '100 GB', memory_type = 'managed');
call gpu_caching("customer", ["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]);

call gpu_processing_resize('150 GB', '20 GB', memory_type = 'managed');
-- Execute query on GPU
call gpu_processing("select
  c_custkey,
  c_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  c_acctbal,
  n_name,
  c_address,
  c_phone,
  c_comment
from
  customer,
  orders,
  lineitem,
  nation
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate >= 19931001
  and o_orderdate <= 19931231
  and l_returnflag = 0
  and c_nationkey = n_nationkey
group by
  c_custkey,
  c_name,
  c_acctbal,
  c_phone,
  n_name,
  c_address,
  c_comment
order by
  revenue desc;");
