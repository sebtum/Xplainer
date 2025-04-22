import os
import tarfile
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def build_tar_index(tar_file_path):
    """
    Build an index of all file members in the tar archive.
    Returns a dictionary mapping member names to a dict with offset and size.
    """
    index = {}
    pbar = tqdm(desc="Processing tar members", unit="member")
    with tarfile.open(tar_file_path, 'r|*') as tar:
        for member in tar:
            pbar.update(1)
            if member.isfile():
                index[member.name] = {
                    'offset': member.offset_data,
                    'size': member.size
                }
    return index

def save_index(index, index_file_path):
    with open(index_file_path, "w") as f:
        json.dump(index, f)

def load_index(index_file_path):
    with open(index_file_path, "r") as f:
        index = json.load(f)
    return index

def modify_image_paths(input_csv, tar_index, output_csv, path_header, part):
    """
    Loads a CSV file containing a column 'tar_path' and adds two new columns:
    'offset' and 'size' based on the tar_index mapping.
    Writes the modified DataFrame to output_csv.
    """
    df = pd.read_csv(input_csv)
    
    def get_offset(tar_path):
        tar_path = tar_path.strip()
        if tar_path in tar_index:
            return tar_index[tar_path]['offset']
        else:
            return None

    def get_size(tar_path):
        tar_path = tar_path.strip()
        if tar_path in tar_index:
            return tar_index[tar_path]['size']
        else:
            return None

    # Create the new columns.
    df.loc[:, "offset"] = df[path_header].apply(get_offset)
    df.loc[:, "size"] = df[path_header].apply(get_size)
    if path_header == "member_path":
        df_filtered = df[df["tar_path"].str.contains(part)]
    df_filtered.to_csv(output_csv, index=False)
    print(f"Modified CSV saved to {output_csv}")

def main():
    # Define paths (update these to match your setup)
    root_dir = Path("/data")
    #tar_file_path = root_dir / "dataset/MIMIC_CXR/images-2.0.0.tar"
    #input_csv = root_dir / "geraugi/plural/pre_processed_data/xplainer_dataset.csv"
    #output_csv = root_dir / "geraugi/plural/pre_processed_data/xplainer_mimic_dataset.csv"
    #index_file = root_dir / "geraugi/plural/dataset_files/mimic_tar_indexes.json"
    #path_header = "tar_path"
    for i in range(8):
        part = f"part{i}"
        tar_file_path = root_dir / f"geraugi/plural/dataset_files/resized_images-2.0.0.{part}.tar"
        input_csv = root_dir / "geraugi/plural/dataset_files/500_xplainer_mimic_dataset.csv"
        output_csv = root_dir / f"geraugi/plural/dataset_files/500_xplainer_mimic_dataset_{part}.csv"
        index_file = root_dir / f"geraugi/plural/dataset_files/resized_mimic_tar_indexes_{part}.json"
        path_header = "member_path" ## FIX ME: used for part
        # If the index file does not exist, build it and save it.
        if not index_file.exists():
            print(f"Building tar index {i}...")
            tar_index = build_tar_index(tar_file_path)
            save_index(tar_index, index_file)
        else:
            print(f"Loading existing tar index {i}...")
            tar_index = load_index(index_file)
    
        modify_image_paths(input_csv, tar_index, output_csv, path_header, part)


if __name__ == "__main__":
    main()
