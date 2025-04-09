import pandas as pd
import torch
from torch.utils.data import Dataset
import ast

class MimicDataset(Dataset):
    def __init__(self, csv_file='/data/geraugi/plural/dataset_files/500_xplainer_mimic_dataset.csv'):
        # Read the CSV file containing dicom_id, xplainer_diseases, and image_path.
        self.data = pd.read_csv(csv_file)
        
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
        image_path = row['tar_path']
        # Get the dictionary of disease labels.
        disease_dict = row['disease_vector']
        # Build a multi-hot vector using the defined order.
        # If a disease is missing in the dictionary, default to 0.
        labels = [disease_dict.get(key, 0) for key in self.final_keys]
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        return image_path, labels_tensor, self.final_keys

# Example usage:
# dataset = XplainerDataset('path_to_your_file.csv')
# img_path, labels, keys = dataset[0]
# print("Image Path:", img_path)
# print("Labels:", labels)
# print("Disease Keys:", keys)
