DROP TABLE IF EXISTS SPAM;

--CREATE TABLE IF NOT EXISTS SPAM 
--  AS FROM read_csv('/home/abigale/all_datasets/wikipedia/wiki_100000.tbl', 
--  header = false, 
--  columns = { 'doc_text' : 'VARCHAR' }, 
--  ignore_errors = True);
--CREATE TABLE IF NOT EXISTS SPAM 
--  AS FROM read_csv('/home/abigale/all_datasets/wikipedia/wiki_500000.tbl', 
--  header = false, 
--  columns = { 'doc_text' : 'VARCHAR' }, 
--  ignore_errors = True);
--CREATE TABLE IF NOT EXISTS SPAM 
--  AS FROM read_csv('/home/abigale/all_datasets/wikipedia/wiki_1000000.tbl', 
--  header = false, 
--  columns = { 'doc_text' : 'VARCHAR' }, 
--  ignore_errors = True);
CREATE TABLE IF NOT EXISTS SPAM 
  AS FROM read_csv('/home/abigale/all_datasets/wikipedia/wiki_5000000.tbl', 
  header = false, 
  columns = { 'doc_text' : 'VARCHAR' }, 
  ignore_errors = True);

call gpu_buffer_init("17 GB", "17 GB");

.timer on

call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(doc_text, 'probabilistic')");