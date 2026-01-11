-- TPC-H Query 18 - GPU Caching and Processing

-- Cache columns for each table
call gpu_caching("customer", ["c_custkey", "c_name"]);
call gpu_caching("orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"]);
call gpu_caching("lineitem", ["l_orderkey", "l_quantity"]);

-- Execute query on GPU
call gpu_processing("select
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice,
  sum(l_quantity)
from
  customer,
  orders,
  lineitem
where
  o_orderkey in (
    select
      l_orderkey
    from
      lineitem
    group by
      l_orderkey
    having
      sum(l_quantity) > 300
    )
  and c_custkey = o_custkey
  and o_orderkey = l_orderkey
group by
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice
order by
  o_totalprice desc,
  o_orderdate,
  o_orderkey;");
