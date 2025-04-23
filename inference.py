import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

from chestxray14 import ChestXray14Dataset
from chexpert import CheXpertDataset
from mimic_dataset import MimicDataset
from descriptors import disease_descriptors_chexpert, disease_descriptors_chestxray14, disease_descriptors_mimic
from model import InferenceModel
from utils import calculate_auroc
import logging
import os
import datetime

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(
        format="%(asctime)s %(message)s", 
        datefmt="%H:%M:%S",
        level=logging.INFO
    )

def inference_chexpert():
    split = 'test'
    dataset = CheXpertDataset(f'data/chexpert/{split}_labels.csv')  # also do test
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=0)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_chexpert)

    all_labels = []
    all_probs_neg = []

    for batch in tqdm(dataloader):
        batch = batch[0]
        image_paths, labels, keys = batch
        image_paths = [Path(image_path) for image_path in image_paths]
        agg_probs = []
        agg_negative_probs = []
        for image_path in image_paths:
            probs, negative_probs = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
            agg_probs.append(probs)
            agg_negative_probs.append(negative_probs)
        probs = {}  # Aggregated
        negative_probs = {}  # Aggregated
        for key in agg_probs[0].keys():
            probs[key] = sum([p[key] for p in agg_probs]) / len(agg_probs)  # Mean Aggregation

        for key in agg_negative_probs[0].keys():
            negative_probs[key] = sum([p[key] for p in agg_negative_probs]) / len(agg_negative_probs)  # Mean Aggregation

        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_chexpert, pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(disease_descriptors_chexpert,
                                                                                                   disease_probs=disease_probs,
                                                                                                   negative_disease_probs=negative_disease_probs,
                                                                                                   keys=keys)
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    # evaluation
    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


def inference_chestxray14():
    dataset = ChestXray14Dataset(f'data/chestxray14/Data_Entry_2017_v2020_modified.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_chestxray14)

    all_labels = []
    all_probs_neg = []
    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        probs, negative_probs = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_chestxray14, pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(disease_descriptors_chestxray14,
                                                                                                   disease_probs=disease_probs,
                                                                                                   negative_disease_probs=negative_disease_probs,
                                                                                                   keys=keys)
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')

def inference_mimic(part,ref=False):
    
    # Create the dataset from your CSV file containing dicom_id, xplainer diseases, and image path.
    if ref == True:
        dataset_path = f'/data/geraugi/plural/dataset_files/correct_500_ref_xplainer_mimic_dataset.csv'
        cache_file = f'/data/geraugi/plural/dataset_files/ref_image_embeddings_cache.csv'
        logging.info("Special case: Using remaining reference images for inference")
    else:
        dataset_path = f'/data/geraugi/plural/dataset_files/500_xplainer_mimic_dataset_part{part}.csv'
        cache_file = f'/data/geraugi/plural/dataset_files/image_embeddings_cache_part{part}.csv'
    logging.info(f"Starting inference on mimic dataset {dataset_path}")
    dataset = MimicDataset(dataset_path)  # Your CSV file for test split
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=0)
    
    inference_model = InferenceModel()
    # Get all disease descriptors from the model using your MIMIC-specific descriptors.
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_mimic)

    all_labels = []
    all_probs_neg = []
    all_dicom_ids = []  # to store dicom_id for each sample
    all_predicted_diseases = []
    all_neg_probs_neg = []
    desc_prob_rows = []  # will hold one dict per (dicom_id, descriptor) pair
    all_image_embeddings = []

    interval = 600
    total   = len(dataloader)
    use_cache_file = False
    cached_image_embeddings_df = None
    if os.path.isfile(cache_file):
        cached_image_embeddings_df = pd.read_csv(cache_file, index_col='dicom_id')
        row_count = len(cached_image_embeddings_df)
        if row_count > 300: # if it is under 300 is most likely testing
            logging.info(f"Using image embedding cache file {cache_file} to speed up inference")
            use_cache_file = True

    for i, batch in enumerate(dataloader, start=1):
        # Since batch is a list with one item, we get the first (and only) element.
        batch = batch[0]
        image_path, labels, keys = batch

        if hasattr(image_path, "member_path"):
            dicom_id = image_path.member_path.replace(".jpg", "")
        else:
            dicom_id = str(image_path).replace(".jpg", "")
        all_dicom_ids.append(dicom_id)
        
        probs, negative_probs, image_embedding = inference_model.get_descriptor_probs(dicom_id, image_path, all_descriptors, use_cache_file=use_cache_file, cached_image_embeddings_df=cached_image_embeddings_df )

        for desc, p in probs.items():
            n = negative_probs.get(desc, None)
            desc_prob_rows.append({
                'dicom_id': dicom_id,
                'descriptor': desc,
                'prob': float(p),       # ensure JSON‑serializable / native type
                'neg_prob': float(n)    # same here
            })
        
        # Get disease-level probabilities using the MIMIC descriptors.
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(
            disease_descriptors_mimic, pos_probs=probs, negative_probs=negative_probs)
        
        # Generate predictions using binary prompting.
        predicted_diseases, prob_vector_neg_prompt, neg_prob_vector_neg_prompt= inference_model.get_predictions_bin_prompting(
            disease_descriptors_mimic, disease_probs=disease_probs,
            negative_disease_probs=negative_disease_probs, keys=keys)

        all_predicted_diseases.append(predicted_diseases)
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        all_neg_probs_neg.append(neg_prob_vector_neg_prompt)
        if use_cache_file == False:
            all_image_embeddings.append(image_embedding)
        gc.collect()
        # At every “interval” batches, print a status line:
        if i % interval == 0 or i == total:
            logging.info(f"Processed {i}/{total} — last dicom: {dicom_id}")

    # Construct a dicom_id, descriptor, prob, neg_prob dataframe
    desc_prob_df = pd.DataFrame.from_records(desc_prob_rows)  
    desc_prob_df.set_index('dicom_id', inplace=True)

    # Stack results into tensors
    all_predicted_diseases = torch.stack(all_predicted_diseases)
    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)
    all_neg_probs_neg = torch.stack(all_neg_probs_neg)

    # Filter out diseases that are not present in any ground truth instance.
    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_neg_probs_neg_clean = all_neg_probs_neg[:, existing_mask]
    all_predicted_diseases_clean = all_predicted_diseases[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    # Sum over the 0‑th (sample) dimension to get per‑disease counts:
    label_counts = all_labels_clean.sum(dim=0)              # shape: (D,)
    pred_counts  = all_predicted_diseases_clean.sum(dim=0)  # shape: (D,)
    N = all_labels_clean.shape[0]

    # Convert to Python integers (optional) and print:
    for idx, disease in enumerate(all_keys_clean):
        n_labels = int(label_counts[idx].item())
        n_preds  = int(pred_counts[idx].item())
        random_perf = n_labels/N
        logging.info(f"{disease}: {n_labels} actual, {n_preds} predicted, prevalence (π): {random_perf:.4f}")

    # Evaluate AUROC overall and per disease (using your calculate_auroc function).
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    logging.info(f"AUROC (overall): {overall_auroc:.5f}")
    for idx, key in enumerate(all_keys_clean):
        logging.info(f'{key}: {per_disease_auroc[idx]:.5f}')
   
    # Convert tensors to numpy arrays for DataFrame creation.
    labels_np = all_labels_clean.cpu().numpy()  # (N_samples, D)
    probs_np = all_probs_neg_clean.cpu().numpy()  # (N_samples, D)
    neg_probs_np = all_neg_probs_neg_clean.cpu().numpy()  # (N_samples, D)
    pred_np = all_predicted_diseases_clean.cpu().numpy()

    # Build DataFrames with dicom_ids as row index and diseases as columns.
    df_labels = pd.DataFrame(data=labels_np, index=all_dicom_ids, columns=all_keys_clean)
    df_probs = pd.DataFrame(data=probs_np, index=all_dicom_ids, columns=all_keys_clean)
    df_neg_probs = pd.DataFrame(data=neg_probs_np, index=all_dicom_ids, columns=all_keys_clean)
    df_pred = pd.DataFrame(data=pred_np, index=all_dicom_ids, columns=all_keys_clean)
    if use_cache_file == False:
        np_img_embs = np.stack([t.cpu().numpy() for t in all_image_embeddings], axis=0)
        E = np_img_embs.shape[1]
        col_names = [f'emb_{i}' for i in range(E)]

        # Build the DataFrame with dicom_id as the index
        df_cache = pd.DataFrame(
            data=np_img_embs,
            index=all_dicom_ids,
            columns=col_names
        )
        df_cache.index.name = 'dicom_id'  # ensures the index is named in the CSV

        # 4. Save to CSV in one call—fast via Pandas’ C engine
        logging.info(f'Writing image embeddings to cache file {cache_file}')
        df_cache.to_csv(cache_file, index_label='dicom_id')
        #df_all_img_emb = df = pd.DataFrame({
        #        'dicom_id': all_dicom_ids,
        #        'image_embedding': all_image_embeddings
        #    }, index=all_dicom_ids
        #)


    # Save the labels and probabilities to output CSV files
    output_folder = create_output_folder(ref=ref)
    if ref == True:
        suffix = 'ref'
    else:
        suffix = part
    labels_csv_path = os.path.join(output_folder, f"xp_inf_labels_{suffix}.csv")
    probabilities_csv_path = os.path.join(output_folder, f"xp_inf_probabilities_{suffix}.csv")
    neg_probs_csv_path = os.path.join(output_folder, f"xp_inf_neg_probabilities_{suffix}.csv")
    pred_csv_path = os.path.join(output_folder, f"xp_inf_predictions_{suffix}.csv")
    confusion_path = os.path.join(output_folder, f'confusion_matrix_{suffix}.csv')
    desc_prob_path = os.path.join(output_folder, f'descriptor_probs_long_{suffix}.csv')
     
    df_labels.to_csv(labels_csv_path)
    df_probs.to_csv(probabilities_csv_path)
    df_neg_probs.to_csv(neg_probs_csv_path)
    df_pred.to_csv(pred_csv_path)
    confusion_df = compute_confusion_df(df_pred, df_labels)

    logging.info(f"Disease labels and their probabilities, and descriptor probabilities saved to CSV:\n {labels_csv_path}\n{probabilities_csv_path}\n{neg_probs_csv_path}\n{desc_prob_path}")
    desc_prob_df.to_csv(desc_prob_path)

    confusion_df.to_csv(confusion_path)
    logging.info(f"Confusion matrix saved to CSV: {confusion_path}")

    logging.info("Metrics per disease (positive vs. negative probability comparison):")
    for idx, disease in enumerate(all_keys_clean):
        y_true = all_labels_clean[:, idx].numpy()               # ground‑truth 0/1
        y_pred = all_predicted_diseases_clean[:, idx].numpy()   # predicted  0/1
        y_score  = all_probs_neg_clean[:, idx].numpy() 

        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        avg_prec = average_precision_score(y_true, y_score)

        logging.info(
            f"{disease}:\n"
            f"Precision: {prec:.4f}   "
            f"Recall: {rec:.4f}   "
            f"F1: {f1:.4f}   "
            f"Average Precision: {avg_prec:.4f}   "
        )

def create_output_folder(base_path='/data/geraugi/plural/dataset_files', folder_name='xplainer_inference', ref=False):
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if ref:
        folder_name = f"ref_{folder_name}"
    # Create folder path
    output_folder = os.path.join(base_path, f"{folder_name}_{timestamp}")
    
    # Create folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    return output_folder

def compute_confusion_df(pred_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with the same shape as inputs where each cell is:
      - 'TP' if pred==1 and label==1
      - 'TN' if pred==0 and label==0
      - 'FP' if pred==1 and label==0
      - 'FN' if pred==0 and label==1
    """
    # Ensure the DataFrames align
    assert pred_df.shape == labels_df.shape, "Prediction and label shapes must match"
    
    # Initialize empty DataFrame
    df_conf = pd.DataFrame(index=pred_df.index, columns=pred_df.columns)
    
    # Vectorized assignments via boolean masks
    df_conf[(pred_df == 1) & (labels_df == 1)] = 'TP'
    df_conf[(pred_df == 0) & (labels_df == 0)] = 'TN'
    df_conf[(pred_df == 1) & (labels_df == 0)] = 'FP'
    df_conf[(pred_df == 0) & (labels_df == 1)] = 'FN'
    
    summary_df = pd.DataFrame({
        'TP': ((pred_df == 1) & (labels_df == 1)).sum(axis=0),
        'TN': ((pred_df == 0) & (labels_df == 0)).sum(axis=0),
        'FP': ((pred_df == 1) & (labels_df == 0)).sum(axis=0),
        'FN': ((pred_df == 0) & (labels_df == 1)).sum(axis=0),
    })

    logging.info(f"Confusion summary:\n{summary_df}")
    return df_conf


if __name__ == '__main__':
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic', help='mimic, chexpert or chestxray14')
    parser.add_argument('--part', type=str, default='0', help='0-7')
    parser.add_argument('--ref', action='store_true', help='use remaining reference images')
    args = parser.parse_args()

    if args.dataset == 'chexpert':
        inference_chexpert()
    elif args.dataset == 'chestxray14':
        inference_chestxray14()
    elif args.dataset == 'mimic':
        inference_mimic(args.part, args.ref)
