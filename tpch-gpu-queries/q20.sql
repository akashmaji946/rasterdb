-- TPC-H Query 20 - GPU Caching and Processing

-- Cache columns for each table
call gpu_caching("supplier", ["s_suppkey", "s_name", "s_address", "s_nationkey"]);
call gpu_caching("nation", ["n_nationkey", "n_name"]);
call gpu_caching("partsupp", ["ps_suppkey", "ps_partkey", "ps_availqty"]);
call gpu_caching("part", ["p_partkey", "p_name"]);
call gpu_caching("lineitem", ["l_partkey", "l_suppkey", "l_shipdate", "l_quantity"]);

-- Execute query on GPU
call gpu_processing("select
  s_name,
  s_address
from
  supplier, nation
where
  s_suppkey in (
    select
      ps_suppkey
    from
      partsupp
    where
      ps_partkey in (
        select
          p_partkey
        from
          part
        where
          p_name like 'forest%'
        )
      and ps_availqty > (
        select
          sum(l_quantity) * 0.5
        from
          lineitem
        where
          l_partkey = ps_partkey
          and l_suppkey = ps_suppkey
          and l_shipdate >= 19940101
          and l_shipdate <= 19941231
        )
    )
  and s_nationkey = n_nationkey
  and n_name = 'CANADA'
  order by
    s_name;");
