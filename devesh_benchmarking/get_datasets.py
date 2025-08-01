import boto3
import os
import argparse

USER_DIR = os.path.expanduser("~")
ALL_DATASET_DIR = os.path.join(USER_DIR, "all_datasets") 
datasets_to_download = [
  "wiki_100000.tbl",
  "wiki_1000000.tbl",
  "wiki_500000.tbl",
  "wiki_5000000.tbl"
  #"random_skewed_1000000_12.tbl",
  #"random_skewed_1000000_25.tbl",
  #"random_skewed_100000_50.tbl",
  #"random_skewed_10000_50.tbl"
]

BUCKET_NAME = "devesh-gpu-all-datasets"
INVALID_EXTENSIONS = [".tar"]
def downloader():
    s3_client = boto3.client('s3')
    for key in s3_client.list_objects(Bucket = BUCKET_NAME)['Contents']:
        object_name = key['Key']

        # Make sure is a valid extension
        is_valid_file = True
        for invalid_extension in INVALID_EXTENSIONS:
            if invalid_extension in object_name:
                is_valid_file = False
                break
            
        if not is_valid_file:
            continue

        if object_name in datasets_to_download:
          print(object_name)
          dataset_dir = os.path.join(ALL_DATASET_DIR, "wikipedia")
          save_path = os.path.join(dataset_dir, object_name)
          if not os.path.exists(save_path):
                print("Downloading", object_name, "to", save_path)
                s3_client.download_file(BUCKET_NAME, object_name, save_path)

if __name__ == "__main__":
  downloader()