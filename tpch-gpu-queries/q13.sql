-- TPC-H Query 13 - GPU Caching and Processing
call gpu_buffer_init('180 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("customer", ["c_custkey"]);
call gpu_caching("orders", ["o_orderkey", "o_custkey", "o_comment"]);

call gpu_processing_resize('150 GB', '20 GB', memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  c_count,
  count(*) as custdist
from (
  select
    c_custkey,
    count(o_orderkey) as c_count
  from
    customer left outer join orders on (
      c_custkey = o_custkey
      and o_comment not like '%special%requests%'
    )
  group by
    c_custkey
  ) as c_orders
group by
  c_count
order by
  custdist desc,
  c_count desc;");
