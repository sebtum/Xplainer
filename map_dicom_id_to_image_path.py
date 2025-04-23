import os
import pandas as pd

def main():
    diff_vqa_path = '/data/geraugi/plural/dataset_files'
    qa_file = f'{diff_vqa_path}/mimic_pair_questions.csv'
    mimic_all_file = f'{diff_vqa_path}/mimic_all.csv'
    plural_dataset = f'/data/geraugi/plural/pre_processed_data/compact_context_w_ref_id.csv'
    plural = True

    df_qa = pd.read_csv(qa_file)[['subject_id', 'study_id', 'ref_id']]
    df_mimic_all = pd.read_csv(mimic_all_file)[['study_id', 'dicom_id']]
    if plural == False:
        df_qa = df_qa[df_qa["question_type"] == "difference"]        
        df = pd.merge(df_qa, df_mimic_all, on='study_id', how='left')
        # Configuration - update these paths as needed
        output_csv = '/data/geraugi/plural/pre_processed_data/dicom_to_tar_image_path.csv'  # Where to save the output CSV
    else:
        pl_df = pd.read_csv(plural_dataset)
        pl_df = pl_df[pl_df["ref_id_mask"]==-1]["dicom_id"].drop_duplicates()
        assert pl_df.shape[0] == 330, f"plural df rows: {pl_df.shape[0]}"
        # Filter df_qa to have only subject_id and ref_id, and rename ref_id to study_id
        df_qa_ref = df_qa[['subject_id', 'ref_id']].rename(columns={'ref_id': 'study_id'})
        # Merge the filtered df_qa with df_mimic_all on study_id
        merged_df = pd.merge(df_qa_ref, df_mimic_all, on='study_id')
        # Merge pl_df with the previous output on dicom_id, using a left join
        df = pd.merge(pl_df, merged_df, on='dicom_id', how='left').drop_duplicates()
        output_csv = '/data/geraugi/plural/pre_processed_data/ref_dicom_to_tar_image_path.csv'
    
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