-- TPC-H Query 4 - GPU Caching and Processing
call gpu_buffer_init('120 GB', '85 GB', pinned_memory_size = '120 GB');

-- Cache columns for each table
call gpu_caching("orders", ["o_orderkey", "o_orderdate", "o_orderpriority"]);
call gpu_caching("lineitem", ["l_orderkey", "l_commitdate", "l_receiptdate"]);

call gpu_processing_resize('180 GB', '1 GB', memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  o_orderpriority,
  count(*) as order_count
from
  orders
where
  o_orderdate >= 19930701
  and o_orderdate <= 19930931
  and exists (
    select
      *
    from
      lineitem
    where
      l_orderkey = o_orderkey
      and l_commitdate < l_receiptdate
    )
group by
  o_orderpriority
order by
  o_orderpriority;");
