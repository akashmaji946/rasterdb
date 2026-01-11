-- TPC-H Query 4 - GPU Caching and Processing

-- Cache columns for each table
call gpu_caching("orders", ["o_orderkey", "o_orderdate", "o_orderpriority"]);
call gpu_caching("lineitem", ["l_orderkey", "l_commitdate", "l_receiptdate"]);

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
