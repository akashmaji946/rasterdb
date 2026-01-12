-- TPC-H Query 17 - GPU Caching and Processing)
call gpu_buffer_init('120 GB', '85 GB', pinned_memory_size = '120 GB');
-- Cache columns for each table
call gpu_caching("lineitem", ["l_partkey", "l_extendedprice", "l_quantity"]);
call gpu_caching("part", ["p_partkey", "p_brand", "p_container"]);
call gpu_processing_resize("120 GB", "2 GB", memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  sum(l_extendedprice) / 7.0 as avg_yearly
from
  lineitem,
  part
where
  p_partkey = l_partkey
  and p_brand = 23
  and p_container = 17
  and l_quantity < (
    select
      avg(l_quantity) * 0.2
    from
      lineitem
    where
      l_partkey = p_partkey
  );");
