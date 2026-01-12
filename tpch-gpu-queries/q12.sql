-- TPC-H Query 12 - GPU Caching and Processing
call gpu_buffer_init('180 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("lineitem", ["l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"]);
call gpu_processing_resize('85 GB', '100 GB');
call gpu_caching("orders", ["o_orderkey", "o_orderpriority"]);
call gpu_processing_resize('180 GB', '2 GB', memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  l_shipmode,
  sum(case
    when o_orderpriority = 0
      or o_orderpriority = 1
    then CAST(1 AS DOUBLE)
    else CAST(0 AS DOUBLE)
  end) as high_line_count,
  sum(case
    when o_orderpriority <> 0
      and o_orderpriority <> 1
    then CAST(1 AS DOUBLE)
    else CAST(0 AS DOUBLE)
  end) as low_line_count
from
  orders,
  lineitem
where
  o_orderkey = l_orderkey
  and l_shipmode in (4, 6)
  and l_commitdate < l_receiptdate
  and l_shipdate < l_commitdate
  and l_receiptdate >= 19940101
  and l_receiptdate <= 19941231
group by
  l_shipmode
order by
  l_shipmode;");
