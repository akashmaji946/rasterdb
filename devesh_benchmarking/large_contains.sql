DROP TABLE IF EXISTS SPAM;

--CREATE TABLE IF NOT EXISTS SPAM 
--  AS FROM read_csv('/home/abigale/all_datasets/large_strings/random_skewed_10000_50.tbl', 
--  header = false, 
--  columns = { 'doc_text' : 'VARCHAR' }, 
--  ignore_errors = True);
--CREATE TABLE IF NOT EXISTS SPAM 
--  AS FROM read_csv('/home/abigale/all_datasets/large_strings/random_skewed_100000_50.tbl', 
--  header = false, 
--  columns = { 'doc_text' : 'VARCHAR' }, 
--  ignore_errors = True);
--CREATE TABLE IF NOT EXISTS SPAM 
--  AS FROM read_csv('/home/abigale/all_datasets/large_strings/random_skewed_1000000_12.tbl', 
--  header = false, 
--  columns = { 'doc_text' : 'VARCHAR' }, 
--  ignore_errors = True);
CREATE TABLE IF NOT EXISTS SPAM 
  AS FROM read_csv('/home/abigale/all_datasets/large_strings/random_skewed_1000000_25.tbl', 
  header = false, 
  columns = { 'doc_text' : 'VARCHAR' }, 
  ignore_errors = True);

call gpu_buffer_init("17 GB", "17 GB");

.timer on

call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'Harum Hic Ex At')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'Harum Hic Ex At')");