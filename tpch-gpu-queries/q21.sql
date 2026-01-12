-- TPC-H Query 21 - GPU Caching and Processing
call gpu_buffer_init('180 GB', '85 GB', pinned_memory_size = '150 GB');
-- Cache columns for each table
call gpu_caching("lineitem", ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"]);
call gpu_processing_resize('85 GB', '100 GB');
call gpu_caching("supplier", ["s_suppkey", "s_name", "s_nationkey"]);
call gpu_caching("orders", ["o_orderkey", "o_orderstatus"]);
call gpu_caching("nation", ["n_nationkey", "n_name"]);
call gpu_processing_resize('220 GB', '2 GB', memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  s_name,
  count(*) as numwait
from
  supplier,
  lineitem l1,
  orders,
  nation
where
  s_suppkey = l1.l_suppkey
  and o_orderkey = l1.l_orderkey
  and o_orderstatus = 1
  and l1.l_receiptdate > l1.l_commitdate
  and exists (
    select
      *
    from
      lineitem l2
    where
      l2.l_orderkey = l1.l_orderkey
      and l2.l_suppkey <> l1.l_suppkey
  )
  and not exists (
    select
      *
    from
      lineitem l3
    where
      l3.l_orderkey = l1.l_orderkey
      and l3.l_suppkey <> l1.l_suppkey
      and l3.l_receiptdate > l3.l_commitdate
  )
  and s_nationkey = n_nationkey
  and n_name = 'SAUDI ARABIA'
group by
  s_name
order by
  numwait desc,
  s_name;");
