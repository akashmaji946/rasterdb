-- TPC-H Query 6 - GPU Caching and Processing
call gpu_buffer_init('150 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for lineitem table
call gpu_caching("lineitem", ["l_extendedprice", "l_discount", "l_shipdate", "l_quantity"]);

-- Execute query on GPU
call gpu_processing("select
  sum(l_extendedprice * l_discount) as revenue
from
  lineitem
where
  l_shipdate >= 19940101
  and l_shipdate <= 19941231
  and l_discount between 0.05 and 0.07
  and l_quantity < 24;");
