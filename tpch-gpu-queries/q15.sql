-- TPC-H Query 15 - GPU Caching and Processing
call gpu_buffer_init('150 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("lineitem", ["l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"]);
call gpu_caching("supplier", ["s_suppkey", "s_name", "s_address", "s_phone"]);

-- Execute query on GPU
call gpu_processing("with revenue_view as (
  select
    l_suppkey as supplier_no,
    sum(l_extendedprice * (1 - l_discount)) as total_revenue
  from
    lineitem
  where
    l_shipdate >= 19960101
    and l_shipdate <= 19960331
  group by
    l_suppkey
)

select
  s_suppkey,
  total_revenue
from
  supplier,
  revenue_view
where
  s_suppkey = supplier_no
  and total_revenue = (
    select
      max(total_revenue)
    from
      revenue_view
    )
order by
  s_suppkey;");
