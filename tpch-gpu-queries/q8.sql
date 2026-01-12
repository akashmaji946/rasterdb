-- TPC-H Query 8 - GPU Caching and Processing
call gpu_buffer_init('180 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("lineitem", ["l_partkey", "l_suppkey", "l_orderkey", "l_extendedprice", "l_discount"]);
call gpu_processing_resize('85 GB', '100 GB');
call gpu_caching("part", ["p_partkey", "p_type"]);
call gpu_caching("supplier", ["s_suppkey", "s_nationkey"]);
call gpu_caching("orders", ["o_orderkey", "o_custkey", "o_orderdate"]);
call gpu_caching("customer", ["c_custkey", "c_nationkey"]);
call gpu_caching("nation", ["n_nationkey", "n_name", "n_regionkey"]);
call gpu_caching("region", ["r_regionkey", "r_name"]);
call gpu_processing_resize('180 GB', '2 GB', memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  o_year,
  sum(case
    when nation = 1
    then volume
    else 0
  end) / sum(volume) as mkt_share
from (
  select
    o_orderdate//10000 as o_year,
    l_extendedprice * (1 - l_discount) as volume,
    n2.n_nationkey as nation
  from
    part,
    supplier,
    lineitem,
    orders,
    customer,
    nation n1,
    nation n2,
    region
  where
    p_partkey = l_partkey
    and s_suppkey = l_suppkey
    and l_orderkey = o_orderkey
    and o_custkey = c_custkey
    and c_nationkey = n1.n_nationkey
    and n1.n_regionkey = r_regionkey
    and r_regionkey = 1
    and s_nationkey = n2.n_nationkey
    and o_orderdate between 19950101 and 19961231
    and p_type = 103
  ) as all_nations
group by
  o_year
order by
  o_year;");
