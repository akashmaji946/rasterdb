-- TPC-H Query 2 - GPU Caching and Processing
call gpu_buffer_init('160 GB', '85 GB', pinned_memory_size = '160 GB');

-- Cache columns for each table
call gpu_caching("part", ["p_partkey", "p_mfgr", "p_size", "p_type"]);
call gpu_caching("supplier", ["s_suppkey", "s_name", "s_address", "s_phone", "s_acctbal", "s_comment", "s_nationkey"]);
call gpu_caching("partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost"]);
call gpu_caching("nation", ["n_nationkey", "n_name", "n_regionkey"]);
call gpu_caching("region", ["r_regionkey", "r_name"]);

-- Execute query on GPU
call gpu_processing("select
  s_acctbal,
  s_name,
  n_name,
  p_partkey,
  p_mfgr,
  s_address,
  s_phone,
  s_comment
from
  part,
  supplier,
  partsupp,
  nation,
  region
where
  p_partkey = ps_partkey
  and s_suppkey = ps_suppkey
  and p_size = 15
  and (p_type + 3) % 5 = 0
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'EUROPE'
  and ps_supplycost = (
    select
      min(ps_supplycost)
    from
      partsupp,
      supplier,
      nation,
      region
    where
      p_partkey = ps_partkey
      and s_suppkey = ps_suppkey
      and s_nationkey = n_nationkey
      and n_regionkey = r_regionkey
      and r_name = 'EUROPE'
    )
order by
  s_acctbal desc,
  n_name,
  s_name,
  p_partkey");
