import pandas as pd
import ast
def main():
    # list of 31 disease names (the order corresponds to positions in the multi-hot vector)
    plural_diseases = [
        'atelectasis', 'pleural effusion', 'lung opacity', 'edema', 
        'cardiomegaly', 'pneumonia', 'consolidation', 'vascular congestion', 
        'pneumothorax', 'infection', 'calcification', 'fracture', 'emphysema',
        'enlargement of the cardiac silhouette', 'pleural thickening', 
        'blunting of the costophrenic angle', 'hernia', 'scoliosis', 
        'heart failure', 'granuloma', 'pneumomediastinum', 'air collection', 
        'hilar congestion', 'hematoma', 'contusion', 'tortuosity of the thoracic aorta', 
        'gastric distention', 'hypoxemia', 'tortuosity of the descending aorta', 
        'hypertensive heart disease', 'thymoma'
    ]

    # List of 13 diseases of interest
    diseases_of_interest = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Emphysema",
        "Fracture", "Hernia", "Lung Opacity", "Pleural Effusion", "Pleural Thickening",
        "Pneumonia", "Pneumothorax"
    ]

    final_keys = diseases_of_interest + ["No Findings"]

    # Determine the indices for the diseases of interest in the full list
    indices = [plural_diseases.index(d.lower()) for d in diseases_of_interest if d.lower() in plural_diseases]

    # Create a new column 'filtered_vector' with the 13-disease multi-hot vector.
    aux_df = pd.read_csv("/data/geraugi/plural/pre_processed_data/compact_context_w_ref_id.csv")
    df_labels = aux_df.loc[:, ['dicom_id', 'disease_vector']].drop_duplicates(subset=['dicom_id'])

    # Convert the string representation of the list into an actual list
    df_labels.loc[:, 'disease_vector'] = df_labels['disease_vector'].apply(ast.literal_eval)

    # Define a function to filter the 31-disease vector to 13 diseases and append a "no findings" label.
    def filter_vector(vec):
        # Extract the 13-disease vector based on the indices
        filtered = [vec[i] for i in indices]
        # If all 13 values are 0, then add a "no findings" label with value 1; otherwise, 0.
        no_findings = 1 if sum(filtered) == 0 else 0
        return filtered + [no_findings]

    # Create a new column with the filtered 13-disease vector plus the no findings label (total length 14)
    df_labels.loc[:, 'xplainer_diseases'] = df_labels['disease_vector'].apply(filter_vector)

    # Now you have another DataFrame 'df_images' that contains:
    # 'dicom_id' and 'image_path'
    df_images = pd.read_csv('/data/geraugi/plural/pre_processed_data/dicom_to_tar_image_path.csv')

    # Merge the two DataFrames on 'dicom_id'
    df_merged = pd.merge(df_images, df_labels[['dicom_id', 'xplainer_diseases']], on='dicom_id', how='inner')
    # FIX ME: check if i am missing info
    output_file = '/data/geraugi/plural/pre_processed_data/xplainer_dataset.csv'
    print(f"Creating file for xplainer dataset: {output_file}")
    print(df_merged.head())
    df_merged.to_csv(output_file)

if __name__ == "__main__":
    main()
