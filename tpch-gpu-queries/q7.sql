-- TPC-H Query 7 - GPU Caching and Processing
call gpu_buffer_init('180 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("lineitem", ["l_suppkey", "l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"]);
call gpu_processing_resize('85 GB', '100 GB');
call gpu_caching("orders", ["o_orderkey", "o_custkey"]);
call gpu_caching("customer", ["c_custkey", "c_nationkey"]);
call gpu_caching("nation", ["n_nationkey", "n_name"]);
call gpu_caching("supplier", ["s_suppkey", "s_nationkey"]);
call gpu_processing_resize('180 GB', '2 GB', memory_type = 'managed');

-- Execute query on GPU
call gpu_processing("select
  supp_nation,
  cust_nation,
  l_year,
  sum(volume) as revenue
from (
  select
    n1.n_name as supp_nation,
    n2.n_name as cust_nation,
    l_shipdate//10000 as l_year,
    l_extendedprice * (1 - l_discount) as volume
  from
    supplier,
    lineitem,
    orders,
    customer,
    nation n1,
    nation n2
  where
    s_suppkey = l_suppkey
    and o_orderkey = l_orderkey
    and c_custkey = o_custkey
    and s_nationkey = n1.n_nationkey
    and c_nationkey = n2.n_nationkey
    and (
      (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
      or (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
    )
    and l_shipdate between 19950101 and 19961231
  ) as shipping
group by
  supp_nation,
  cust_nation,
  l_year
order by
  supp_nation,
  cust_nation,
  l_year;");
