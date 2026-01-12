-- TPC-H Query 14 - GPU Caching and Processing
call gpu_buffer_init('180 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("lineitem", ["l_partkey", "l_extendedprice", "l_discount", "l_shipdate"]);
call gpu_caching("part", ["p_partkey", "p_type"]);

-- Execute query on GPU
call gpu_processing("select
    sum(case
    when (p_type >= 125 and p_type < 150)
    then l_extendedprice * (1 - l_discount)
    else 0.0
    end) * 100.0 / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
  lineitem,
  part
where
  l_partkey = p_partkey
  and l_shipdate >= 19950901
  and l_shipdate <= 19950931;");
