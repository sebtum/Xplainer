import os
import pandas as pd

def main():
    diff_vqa_path = '/data/geraugi/plural/dataset_files'
    qa_file = f'{diff_vqa_path}/mimic_pair_questions.csv'
    mimic_all_file = f'{diff_vqa_path}/mimic_all.csv'

    df_qa = pd.read_csv(qa_file)
    df_qa = df_qa[df_qa["question_type"] == "difference"]\
        [['subject_id', 'study_id', 'ref_id']]
    df_mimic_all = pd.read_csv(mimic_all_file)[['study_id', 'dicom_id']]
    df = pd.merge(df_qa, df_mimic_all, on='study_id', how='left')
    # Configuration - update these paths as needed
    output_csv = '/data/geraugi/plural/pre_processed_data/dicom_to_tar_image_path.csv'  # Where to save the output CSV
    base_tar_path = 'images-2.0.0'  # Base path inside the tar archive

    # Ensure all IDs are strings for proper path construction
    df['subject_id'] = df['subject_id'].astype(str)
    df['study_id'] = df['study_id'].astype(str)
    df['dicom_id'] = df['dicom_id'].astype(str)



    # Function to build the tar image path using subject and study IDs along with the dicom id
    def build_tar_path(row):
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        # Construct the internal tar path: base_tar_path/pXX/p<subject_id>/s<study_id>/<dicom_id>.jpg
        return os.path.join(
            base_tar_path,
            'p' + subject_id[:2],
            'p' + subject_id,
            's' + study_id,
            dicom_id + '.jpg'
        )

    # Build the tar path for each row
    df['tar_path'] = df.apply(build_tar_path, axis=1)

    # Output CSV with just dicom_id and tar_path columns
    output_df = df[['dicom_id', 'tar_path']]
    print(output_df.head())
    output_df.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")

if __name__ == "__main__":
    main()