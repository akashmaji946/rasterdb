-- TPC-H Query 9 - GPU Caching and Processing

-- Cache columns for each table
call gpu_caching("part", ["p_partkey", "p_name"]);
call gpu_caching("supplier", ["s_suppkey", "s_nationkey"]);
call gpu_caching("lineitem", ["l_suppkey", "l_partkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity"]);
call gpu_caching("partsupp", ["ps_suppkey", "ps_partkey", "ps_supplycost"]);
call gpu_caching("orders", ["o_orderkey", "o_orderdate"]);
call gpu_caching("nation", ["n_nationkey", "n_name"]);

-- Execute query on GPU
call gpu_processing("select
  nation,
  o_year,
  sum(amount) as sum_profit
from(
  select
    n_name as nation,
    o_orderdate//10000 as o_year,
    l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
  from
    part,
    supplier,
    lineitem,
    partsupp,
    orders,
    nation
  where
    s_suppkey = l_suppkey
    and ps_suppkey = l_suppkey
    and ps_partkey = l_partkey
    and p_partkey = l_partkey
    and o_orderkey = l_orderkey
    and s_nationkey = n_nationkey
    and p_name like '%green%'
  ) as profit
group by
  nation,
  o_year
order by
  nation,
  o_year desc;");
