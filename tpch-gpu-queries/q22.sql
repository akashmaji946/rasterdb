-- TPC-H Query 22 - GPU Caching and Processing
call gpu_buffer_init('150 GB', '85 GB', pinned_memory_size = '150 GB');

-- Cache columns for each table
call gpu_caching("orders", ["o_custkey"]);
call gpu_caching("customer", ["c_custkey", "c_phone", "c_acctbal"]);

-- Execute query on GPU
call gpu_processing("select
    cntrycode,
    count(*) as numcust,
    sum(c_acctbal) as totacctbal
from
    (
        select
            substring(c_phone from 1 for 2) as cntrycode,
            c_acctbal
        from
            customer c
        where
            substring(c_phone from 1 for 2) in
                ('24', '31', '11', '16', '21', '20', '34')
            and c_acctbal > (
                select
                    avg(c_acctbal)
                from
                    customer
                where
                    c_acctbal > 0.00
                    and substring(c_phone from 1 for 2) in
                        ('24', '31', '11', '16', '21', '20', '34')
            )
            and not exists (
                select
                    *
                from
                    orders o
                where
                    o.o_custkey = c.c_custkey
            )
    ) as custsale
group by
    cntrycode
order by
    cntrycode;");
