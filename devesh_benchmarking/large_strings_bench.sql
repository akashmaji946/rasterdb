DROP TABLE IF EXISTS SPAM;

CREATE TABLE SPAM (
  P_VAL VARCHAR(256)
);

COPY SPAM from '/mnt/wiscdb/abigale/string_dataset_csvs/large_strings_dataset.csv';

call gpu_buffer_init("17 GB", "17 GB");

.timer on

--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%Lorem ipsum%Harum Hic Ex At%'");

call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'Harum Hic Ex At')");