-- TPC-H Query 3 - GPU Caching and Processing
call gpu_buffer_init('150 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("customer", ["c_custkey", "c_mktsegment"]);
call gpu_caching("orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"]);
call gpu_caching("lineitem", ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"]);

call gpu_processing_resize('180 GB', '10 GB', memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  l_orderkey,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  o_orderdate,
  o_shippriority
from
  customer,
  orders,
  lineitem
where
  c_mktsegment = 1
  and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < 19950315
  and l_shipdate > 19950315
group by
  l_orderkey,
  o_orderdate,
  o_shippriority
order by
  revenue desc,
  o_orderdate
limit 10;");
