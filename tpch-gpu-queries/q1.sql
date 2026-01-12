-- TPC-H Query 1 - GPU Caching and Processing
call gpu_buffer_init('165 GB', '85 GB', pinned_memory_size = '165 GB');

-- Cache columns for lineitem table
call gpu_caching("lineitem", ["l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_shipdate"]);

call gpu_processing_resize('120 GB', '1 GB', memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  l_returnflag,
  l_linestatus,
  sum(l_quantity) as sum_qty,
  sum(l_extendedprice) as sum_base_price,
  sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
  sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
  avg(l_quantity) as avg_qty,
  avg(l_extendedprice) as avg_price,
  avg(l_discount) as avg_disc,
  count(*) as count_order
from
  lineitem
where
  l_shipdate <= 19920902
group by
  l_returnflag,
  l_linestatus
order by
  l_returnflag,
  l_linestatus;");
