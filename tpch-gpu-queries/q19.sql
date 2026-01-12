-- TPC-H Query 19 - GPU Caching and Processing
call gpu_buffer_init('180 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("lineitem", ["l_partkey", "l_extendedprice", "l_discount", "l_quantity", "l_shipmode", "l_shipinstruct"]);
call gpu_processing_resize('85 GB', '100 GB');
call gpu_caching("part", ["p_partkey", "p_brand", "p_container", "p_size"]);

call gpu_processing_resize('85 GB', '2 GB');

-- Execute query on GPU
call gpu_processing("select
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  lineitem,
  part
where
  p_partkey = l_partkey
  and (
    (
      p_brand = 12
      and p_container in (0, 1, 4, 5)
      and l_quantity >= 1 and l_quantity <= 11
      and p_size between 1 and 5
      and l_shipmode in (0, 1)
      and l_shipinstruct = 0
    )
    or
    (
      p_brand = 23
      and p_container in (17, 18, 20, 21)
      and l_quantity >= 10 and l_quantity <= 20
      and p_size between 1 and 10
      and l_shipmode in (0, 1)
      and l_shipinstruct = 0
    )
    or
    (
      p_brand = 34
      and p_container in (8, 9, 12, 13)
      and l_quantity >= 20 and l_quantity <= 30
      and p_size between 1 and 15
      and l_shipmode in (0, 1)
      and l_shipinstruct = 0
    )
  );");
