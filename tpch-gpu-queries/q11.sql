-- TPC-H Query 11 - GPU Caching and Processing

-- Cache columns for each table
call gpu_caching("partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"]);
call gpu_caching("supplier", ["s_suppkey", "s_nationkey"]);
call gpu_caching("nation", ["n_nationkey", "n_name"]);

-- Execute query on GPU
call gpu_processing("select
  *
from (
  select
    ps_partkey,
    sum(ps_supplycost * ps_availqty) as value
  from
    partsupp,
    supplier,
    nation
  where
    ps_suppkey = s_suppkey
    and s_nationkey = n_nationkey
    and n_name = 'GERMANY'
  group by
    ps_partkey
) as inner_query
where
  value > (
    select
      sum(ps_supplycost * ps_availqty) * 0.0000000333
    from
      partsupp,
      supplier,
      nation
    where
      ps_suppkey = s_suppkey
      and s_nationkey = n_nationkey
      and n_name = 'GERMANY'
  )
order by
  value desc,
  ps_partkey;");
