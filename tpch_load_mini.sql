DROP TABLE IF EXISTS nation;
DROP TABLE IF EXISTS region;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS partsupp;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS lineitem;

CREATE TABLE nation  ( n_nationkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                       n_name       CHAR(25) NOT NULL,
                       n_regionkey  INTEGER NOT NULL,
                       n_comment    VARCHAR(152));

CREATE TABLE region  ( r_regionkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                       r_name       CHAR(25) NOT NULL,
                       r_comment    VARCHAR(152));

CREATE TABLE part  ( p_partkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                     p_name        VARCHAR(55) NOT NULL,
                     p_mfgr        INTEGER NOT NULL,
                     p_brand       INTEGER NOT NULL,
                     p_type        INTEGER NOT NULL,
                     p_size        INTEGER NOT NULL,
                     p_container   INTEGER NOT NULL,
                     p_retailprice FLOAT NOT NULL,
                     p_comment     VARCHAR(23) NOT NULL );

CREATE TABLE supplier ( s_suppkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        s_name        CHAR(25) NOT NULL,
                        s_address     VARCHAR(40) NOT NULL,
                        s_nationkey   INTEGER NOT NULL,
                        s_phone       CHAR(15) NOT NULL,
                        s_acctbal     FLOAT NOT NULL,
                        s_comment     VARCHAR(101) NOT NULL);

CREATE TABLE partsupp ( ps_partkey     INTEGER NOT NULL,
                        ps_suppkey     INTEGER NOT NULL,
                        ps_availqty    INTEGER NOT NULL,
                        ps_supplycost  FLOAT NOT NULL,
                        ps_comment     VARCHAR(199) NOT NULL,
                        CONSTRAINT PS_PARTSUPPKEY UNIQUE(PS_PARTKEY, PS_SUPPKEY) );

CREATE TABLE customer ( c_custkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        c_name        VARCHAR(25) NOT NULL,
                        c_address     VARCHAR(40) NOT NULL,
                        c_nationkey   INTEGER NOT NULL,
                        c_phone       CHAR(15) NOT NULL,
                        c_acctbal     FLOAT NOT NULL,
                        c_mktsegment  INTEGER NOT NULL,
                        c_comment     VARCHAR(117) NOT NULL);

CREATE TABLE orders  ( o_orderkey       BIGINT NOT NULL UNIQUE PRIMARY KEY,
                       o_custkey        INTEGER NOT NULL,
                       o_orderstatus    INTEGER NOT NULL,
                       o_totalprice     FLOAT NOT NULL,
                       o_orderdate      INTEGER NOT NULL,
                       o_orderpriority  INTEGER NOT NULL,
                       o_clerk          INTEGER NOT NULL,
                       o_shippriority   INTEGER NOT NULL,
                       o_comment        VARCHAR(79) NOT NULL);

CREATE TABLE lineitem ( l_orderkey    BIGINT NOT NULL,
                        l_partkey     INTEGER NOT NULL,
                        l_suppkey     INTEGER NOT NULL,
                        l_linenumber  INTEGER NOT NULL,
                        l_quantity    FLOAT NOT NULL,
                        l_extendedprice  FLOAT NOT NULL,
                        l_discount    FLOAT NOT NULL,
                        l_tax         FLOAT NOT NULL,
                        l_returnflag  INTEGER NOT NULL,
                        l_linestatus  INTEGER NOT NULL,
                        l_shipdate    INTEGER NOT NULL,
                        l_commitdate  INTEGER NOT NULL,
                        l_receiptdate INTEGER NOT NULL,
                        l_shipinstruct INTEGER NOT NULL,
                        l_shipmode     INTEGER NOT NULL,
                        l_comment      VARCHAR(44) NOT NULL);

COPY lineitem FROM 'test_datasets/tpch-mod-dbgen/s1/lineitem.tbl' WITH (HEADER false, DELIMITER '|');
COPY orders FROM 'test_datasets/tpch-mod-dbgen/s1/orders.tbl' WITH (HEADER false, DELIMITER '|');
COPY supplier FROM 'test_datasets/tpch-mod-dbgen/s1/supplier.tbl' WITH (HEADER false, DELIMITER '|');
COPY part FROM 'test_datasets/tpch-mod-dbgen/s1/part.tbl' WITH (HEADER false, DELIMITER '|');
COPY customer FROM 'test_datasets/tpch-mod-dbgen/s1/customer.tbl' WITH (HEADER false, DELIMITER '|');
COPY partsupp FROM 'test_datasets/tpch-mod-dbgen/s1/partsupp.tbl' WITH (HEADER false, DELIMITER '|');
COPY nation FROM 'test_datasets/tpch-mod-dbgen/s1/nation.tbl' WITH (HEADER false, DELIMITER '|');
COPY region FROM 'test_datasets/tpch-mod-dbgen/s1/region.tbl' WITH (HEADER false, DELIMITER '|');
