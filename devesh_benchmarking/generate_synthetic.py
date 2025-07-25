import os
import csv
from lorem_text import lorem

output_mb = 2048
output_dir = "/mnt/wiscdb/abigale/string_dataset_csvs"
output_filename = str(output_mb) + "_mb_synthetic_dataset.csv"

NORMAL_WORD = 30
LONG_WORD = 250
GIGABYTE = 1073741824
MEGABYTE = 1048576
NORMAL_WORD_NUMBER = 300

output_file = open(output_dir + "/" + output_filename, "w")
output_csv = csv.writer(output_file)
output_csv.writerow(["P_VAL"])

total_bytes = 0
total_len = output_mb * MEGABYTE
while total_bytes < total_len:
  for i in range(0,300):
    word = lorem.words(NORMAL_WORD)
    output_csv.writerow([word])
    total_bytes += len(word)

    if total_bytes >= total_len:
      break

  if total_bytes < total_len:
    long_word = lorem.words(LONG_WORD)
    output_csv.writerow([long_word])
    total_bytes += len(long_word)