import pandas as pd
import matplotlib.pyplot as plt

#csv_path = '/mnt/wiscdb/abigale/string_dataset_csvs/400_mb_synthetic_dataset.csv';
#csv_path = '/mnt/wiscdb/abigale/string_dataset_csvs/800_mb_synthetic_dataset.csv';
#csv_path = '/mnt/wiscdb/abigale/string_dataset_csvs/1200_mb_synthetic_dataset.csv';
#csv_path = '/mnt/wiscdb/abigale/string_dataset_csvs/1600_mb_synthetic_dataset.csv';
#csv_path = '/mnt/wiscdb/abigale/string_dataset_csvs/two_gb_synthetic_dataset.csv';
#csv_path = '/mnt/wiscdb/abigale/string_dataset_csvs/five_gb_synthetic_dataset.csv';
#csv_path = '/mnt/wiscdb/abigale/string_dataset_csvs/medium_synthetic_dataset.csv';
csv_path = "/mnt/wiscdb/abigale/string_dataset_csvs/giant_synthetic_dataset.csv"

df = pd.read_csv(csv_path)
# Extract the first column
values = df.iloc[:, 0].astype(str)

# Compute lengths
lengths = values.str.len()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of String Lengths")
plt.xlabel("String Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.yscale("log")
plt.savefig("help.pdf")