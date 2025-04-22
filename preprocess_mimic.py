import os
import io
import tarfile
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def process_chunk(chunk_df, process_id, input_tar_file, output_tar_file, out_tar_file_base_name):
    """
    Process a subset (chunk) of the CSV rows. For each row, open the image from the input tar,
    resize it and add it to an output tar file unique to this process.
    
    Returns a tuple: (list of index records, path to the partial tar file)
    """
    # Create a process-specific output tar file.
    part_tar_path = output_tar_file.with_name(output_tar_file.stem + f".part{process_id}" + output_tar_file.suffix)
    processed_records = []
    
    # Open the output tar file in write mode.
    with tarfile.open(part_tar_path, "w") as tar_out:
        # Each process opens its own read handle on the input tar.
        with tarfile.open(input_tar_file, 'r') as tar_in:
            # Get the underlying file object for random access.
            f_in = tar_in.fileobj
            for _, row in chunk_df.iterrows():
                dicom_id = row['dicom_id']
                output_filename = f"{dicom_id}.jpg"
                try:
                    offset = int(row['offset'])
                    size = int(row['size'])
                except Exception as e:
                    print(f"Process {process_id}: Error parsing offset/size for {dicom_id}: {e}")
                    continue

                try:
                    f_in.seek(offset)
                    image_data = f_in.read(size)
                except Exception as e:
                    print(f"Process {process_id}: Error reading bytes for {dicom_id} at offset {offset} with size {size}: {e}")
                    continue

                try:
                    img = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    print(f"Process {process_id}: Error opening image for {dicom_id}: {e}")
                    continue

                # Convert to RGB and resize.
                img = img.convert("RGB")
                if img.size[0] < img.size[1]:
                    new_size = (512, int(512 * img.size[1] / img.size[0]))
                else:
                    new_size = (int(512 * img.size[0] / img.size[1]), 512)
                img = img.resize(new_size)

                # Save image into a BytesIO buffer.
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)

                # Get current output tar offset.
                current_offset = tar_out.fileobj.tell()

                # Create a header and add the file.
                tarinfo = tarfile.TarInfo(name=output_filename)
                tarinfo.size = len(buf.getbuffer())
                tar_out.addfile(tarinfo, fileobj=buf)

                # Build a record for the index CSV.
                record = {
                    "dicom_id": dicom_id,
                    "tar_path": part_tar_path,
                    "member_path": output_filename,
                    "offset": current_offset,
                    "size": tarinfo.size,
                    "disease_vector": row["xplainer_diseases"]
                }
                processed_records.append(record)
    return processed_records, part_tar_path

def convert_images_parallel():
    # Define paths (update these to match your environment)
    root_dir = Path("/data")
    csv_file = root_dir / 'geraugi/plural/pre_processed_data/xplainer_mimic_dataset.csv'  # Must contain columns including 'dicom_id', 'offset', 'size', 'xplainer_diseases'
    input_tar_file = root_dir / 'dataset/MIMIC_CXR/images-2.0.0.tar'    # Input tar archive with original images
    output_tar_file = root_dir / 'geraugi/plural/dataset_files/resized_images-2.0.0.tar'  # Final output tar file that will result from merging partial tars
    index_csv_file = root_dir / 'geraugi/plural/dataset_files/500_xplainer_mimic_dataset.csv'

    # Use the base file name (for the "tar_path" field in the index CSV).
    out_tar_file_base_name = os.path.basename(output_tar_file)  # e.g., "resized_images-2.0.0.tar"

    # Load the CSV.
    df = pd.read_csv(csv_file)
    #df = df.head(800) # FIX ME!!

    # If index CSV already exists, load processed dicom_ids and filter out those records.
    processed_records = []
    existing_dicom_ids = set()
    if index_csv_file.exists():
        existing_df = pd.read_csv(index_csv_file)
        processed_records = existing_df.to_dict('records')
        existing_dicom_ids = set(existing_df['dicom_id'].tolist())
        print(f"Loaded {len(existing_dicom_ids)} processed records from index file.")
    else:
        print("No index CSV found; starting from scratch.")
    
    if existing_dicom_ids:
        df = df[~df['dicom_id'].isin(existing_dicom_ids)]
        df.reset_index(drop=True, inplace=True)
        print(f"{len(df)} records remaining after filtering already processed entries.")

    # Split the DataFrame into 8 roughly equal chunks.
    num_processes = 8
    chunks = np.array_split(df, num_processes)

    all_records = []
    part_tar_files = []
    # Use ProcessPoolExecutor to run the chunks in parallel.
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            futures.append(executor.submit(
                process_chunk, chunk, i, input_tar_file, output_tar_file, out_tar_file_base_name))
        for future in tqdm(futures, desc="Processing chunks"):
            records, part_tar_path = future.result()
            all_records.extend(records)
            part_tar_files.append(part_tar_path)

    # Combine any previously processed records.
    all_records.extend(processed_records)
    df_index = pd.DataFrame(all_records)
    df_index.to_csv(index_csv_file, index=False)
    print(f"Index CSV saved to {index_csv_file}")

    print("Partial tar files are kept separately:")
    for part_file in sorted(part_tar_files):
        print(f"  {part_file}")

if __name__ == "__main__":
    # On Windows, the multiprocessing module requires
    # the entry point to be guarded by if __name__=='__main__'.
    convert_images_parallel()
