output_dir = "/mnt/wiscdb/abigale/string_dataset_csvs"
output_filename = "two_gb_synthetic_dataset.csv"
filename = output_dir + "/" + output_filename
longest_str = max(open(filename, 'r'), key=len)
print(len(longest_str))