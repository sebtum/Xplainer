import pandas as pd
import torch
from torch.utils.data import Dataset
import ast
from pathlib import Path
import io

class TarMember:
    """
    A lightweight wrapper to mimic a Path-like object for a file stored as a member in a tar archive.

    This object holds the tar archive location, the internal member path, its byte offset, and size.
    It provides an open() method so that libraries expecting a file-like object (for example, Pillow)
    can operate on it directly.
    """
    def __init__(self, tar_path: Path, member_path: str, offset: int, size: int):
        self.tar_path = Path(tar_path)
        self.member_path = member_path
        self.offset = offset
        self.size = size

    @property
    def suffix(self) -> str:
        # Determine the file extension based on the member name.
        return Path(self.member_path).suffix.lower()

    def open(self, mode="rb"):
        # Open the tar file, seek to the proper offset, and read the member bytes.
        # Note: This works best for uncompressed tar archives.
        with self.tar_path.open(mode) as f:
            f.seek(self.offset)
            data = f.read(self.size)
        if not data.startswith(b'\xff\xd8'):
            raise ValueError(f"Invalid JPEG header for {self.member_path}")
        fileobj = io.BytesIO(data)
        # Set the name attribute so that imageio can infer the file type.
        fileobj.name = self.member_path
        fileobj.seek(0)  # Ensure we're at the start
        return fileobj

    def __hash__(self):
        # This implementation makes TarMember usable as a key in caching dictionaries.
        return hash((str(self.tar_path), self.member_path, self.offset, self.size))

    def __eq__(self, other):
        if not isinstance(other, TarMember):
            return False
        return (self.tar_path, self.member_path, self.offset, self.size) == \
               (other.tar_path, other.member_path, other.offset, other.size)

    def __repr__(self):
        return f"<TarMember {self.member_path} from {self.tar_path} offset: {self.offset} size: {self.size}>"
    
class MimicDataset(Dataset):
    def __init__(self, csv_file='/data/geraugi/plural/dataset_files/500_xplainer_mimic_dataset_part0.csv'):
        # Read the CSV file containing dicom_id, xplainer_diseases, and image_path.
        self.data = pd.read_csv(csv_file)
        #self.data = self.data.head(100) #FIX ME
        
        # Convert the string representation of the dictionary to an actual dictionary.
        # (If the column is already a dictionary, this step is not needed.)
        self.data['disease_vector'] = self.data['disease_vector'].apply(ast.literal_eval)
        
        # Define the final keys (order matters). These are the 13 diseases of interest plus "No Findings".
        self.final_keys = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Emphysema",
            "Fracture", "Hernia", "Lung Opacity", "Pleural Effusion", "Pleural Thickening",
            "Pneumonia", "Pneumothorax", "No Findings"
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Extract metadata for the tar archive member.
        tar_path = row['tar_path']           # e.g., "/data/images/archive.tar"
        member_path = row['member_path']       # e.g., "images/image001.jpg" (member_path)
        offset = int(row['offset'])            # starting byte offset
        size = int(row['size'])                # size in bytes

        # Create the TarMember object using the provided metadata.
        tar_member = TarMember(tar_path=Path(tar_path),
                               member_path=member_path,
                               offset=offset,
                               size=size)

        # Build a multi-hot label vector in a fixed order.
        labels_tensor = torch.tensor(row['disease_vector'], dtype=torch.float)

        return tar_member, labels_tensor, self.final_keys

# Example usage:
# dataset = XplainerDataset('path_to_your_file.csv')
# img_path, labels, keys = dataset[0]
# print("Image Path:", img_path)
# print("Labels:", labels)
# print("Disease Keys:", keys)
