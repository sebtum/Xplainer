import pandas as pd
from pathlib import Path
from PIL import Image
import tarfile
import io
from tqdm import tqdm
import os

def convert_images():
    # Define paths (update these to match your environment)
    root_dir = Path("/data")
    csv_file = root_dir / 'geraugi/plural/pre_processed_data/xplainer_mimic_dataset.csv' # CSV with columns: 'dicom_id' and 'tar_path'
    input_tar_file = root_dir / 'dataset/MIMIC_CXR/images-2.0.0.tar'    # Tar archive containing the original images
    output_tar_file = root_dir / 'geraugi/plural/dataset_files/resized_images-2.0.0.tar'  # Output tar file for resized images
    index_csv_file = root_dir / 'geraugi/plural/dataset_files/500_xplainer_mimic_dataset.csv'
    # Extract the base file name
    out_tar_file_name = os.path.basename(output_tar_file)  # "resized_images-2.0.0.tar"
    out_tar_file_base_name, ext = os.path.splitext(out_tar_file_name) 

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Decide the mode for the output tar file:
    # If the file exists and its size is > 0, use append ("a"); otherwise, write ("w").
    if output_tar_file.exists() and output_tar_file.stat().st_size > 0:
        mode = "a"
    else:
        mode = "w"

    # Create a list to store records for the new resized images.
    processed_records = []
    existing_dicom_ids = set()
    if index_csv_file.exists():
        existing_df = pd.read_csv(index_csv_file)
        processed_records = existing_df.to_dict('records')
        existing_dicom_ids = set(existing_df['dicom_id'].tolist())
        print(f"Loaded {len(existing_dicom_ids)} processed records from index file.")
    else:
        print("No index CSV found; starting from scratch.")
    
    # Open the output tar file
    with tarfile.open(output_tar_file, mode) as tar_out:
        # Open the input tar file once
        with tarfile.open(input_tar_file, 'r') as tar_in:
            # Get the underlying file object for random access.
            f = tar_in.fileobj
            print(f"Processing {input_tar_file}")
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                dicom_id = row['dicom_id']
                # Define the output filename (we use dicom_id.jpg as a unique identifier)
                output_filename = f"{dicom_id}.jpg"
                
                # Check if we've already processed this file by checking our current processed_records.
                if any(record['dicom_id'] == dicom_id for record in processed_records):
                    continue

                try:
                    # Read offset and size information from the CSV.
                    offset = int(row['offset'])
                    size = int(row['size'])
                except Exception as e:
                    print(f"Error parsing offset/size for {dicom_id}: {e}")
                    continue

                # Seek directly to the offset in the tar file and read exactly 'size' bytes.
                try:
                    f.seek(offset)
                    image_data = f.read(size)
                except Exception as e:
                    print(f"Error reading bytes for {dicom_id} at offset {offset} with size {size}: {e}")
                    continue

                # Open the image from the BytesIO buffer
                try:
                    img = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    print(f"Error opening image for {dicom_id}: {e}")
                    continue

                # Convert image to RGB and resize
                img = img.convert("RGB")
                if img.size[0] < img.size[1]:
                    new_size = (512, int(512 * img.size[1] / img.size[0]))
                else:
                    new_size = (int(512 * img.size[0] / img.size[1]), 512)
                img = img.resize(new_size)

                # Save the resized image into a BytesIO buffer
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)

                # Record the current offset in the output tar file.
                current_offset = tar_out.fileobj.tell()

                # Prepare the tar header for the new file.
                tarinfo = tarfile.TarInfo(name=output_filename)
                tarinfo.size = len(buf.getbuffer())
                tar_out.addfile(tarinfo, fileobj=buf)

                # Create a record for this file.
                record = {
                    "dicom_id": dicom_id,
                    "tar_path": f'{out_tar_file_base_name}/{output_filename}',
                    "offset": current_offset,
                    "size": tarinfo.size,
                    "disease_vector": row["xplainer_diseases"]
                }
                processed_records.append(record)

    # Write (or overwrite) the index CSV file with the processed records.
    df_index = pd.DataFrame(processed_records)
    df_index.to_csv(index_csv_file, index=False)
    print(f"Index CSV saved to {index_csv_file}")

if __name__ == "__main__":
    convert_images()
