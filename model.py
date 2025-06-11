from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from health_multimodal.image import get_biovil_resnet_inference
from health_multimodal.text import get_cxr_bert_inference
from health_multimodal.vlp import ImageTextInferenceEngine

from utils import cos_sim_to_prob, prob_to_log_prob, log_prob_to_prob


class InferenceModel():
    def __init__(self):
        self.text_inference = get_cxr_bert_inference()
        self.image_inference = get_biovil_resnet_inference()
        self.image_text_inference = ImageTextInferenceEngine(
            image_inference_engine=self.image_inference,
            text_inference_engine=self.text_inference,
        )
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #FIX ME
        self.device = "cpu"
        self.image_text_inference.to(self.device)

        # caches for faster inference
        self.text_embedding_cache = {}
        self.image_embedding_cache = {}

        self.transform = self.image_inference.transform

    def get_similarity_score_from_raw_data(self, image_embedding, query_text: str, similarity_mode='average') -> float:
        """Compute the cosine similarity score between an image and one or more strings.
        If multiple strings are passed, their embeddings are averaged before L2-normalization.
        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :return: The similarity score between the image and the text.
        """
        assert similarity_mode in ("average", "max"), \
            f"Unknown similarity_mode: {similarity_mode}"
        assert not self.image_text_inference.image_inference_engine.model.training
        assert not self.image_text_inference.text_inference_engine.model.training
        # Normalize query_text input and prepare cache key
        if isinstance(query_text, str):
            descriptors = [query_text]
        else:
            descriptors = query_text
        # For each descriptor, either pull from cache or compute its normalized embedding
        desc_embs = []
        for desc in descriptors:
            if desc in self.text_embedding_cache:
                emb = self.text_embedding_cache[desc]
            else:
                # get a (1, D) tensor, unnormalized
                raw = self.image_text_inference.text_inference_engine.get_embeddings_from_prompt(
                    [desc], normalize=False
                )
                # raw is shape (1, D); squeeze to (D,)
                raw = raw.squeeze(0)
                # L2-normalize
                emb = F.normalize(raw, p=2, dim=0)
                self.text_embedding_cache[desc] = emb
            desc_embs.append(emb)

        # Stack into (N, D)
        embs = torch.stack(desc_embs, dim=0)

        if similarity_mode == "average":
            # mean over descriptors, then renormalize
            mean_emb = embs.mean(dim=0)
            mean_emb = F.normalize(mean_emb, p=2, dim=0)
            cos_sim = image_embedding @ mean_emb

        else:  # max
            # compute N similarities then take max
            sims = embs @ image_embedding  # (N,)
            cos_sim = sims.max()

        return cos_sim.item()

    def process_image(self, image):
        ''' same code as in image_text_inference.image_inference_engine.get_projected_global_embedding() but adapted to deal with image instances instead of path'''

        transformed_image = self.transform(image)
        projected_img_emb = self.image_inference.model.forward(transformed_image).projected_global_embedding
        projected_img_emb = F.normalize(projected_img_emb, dim=-1)
        assert projected_img_emb.shape[0] == 1
        assert projected_img_emb.ndim == 2
        return projected_img_emb[0]

    def get_descriptor_probs(
            self, 
            dicom_id, 
            image_path: Path, 
            descriptors: List[str], 
            do_negative_prompting=True, 
            demo=False, 
            use_cache_file=False, 
            cached_image_embeddings_df=None,
            diagnosis=False,
            ):
        probs = {}
        negative_probs = {}
        if use_cache_file:
            # 1. Get the cached row as a Series
            row = cached_image_embeddings_df.loc[dicom_id]
            if row.empty:
                raise KeyError(f"No entry for dicom_id={dicom_id}")
            # 2. Convert to NumPy and then to tensor
            emb_array     = row.to_numpy()         
            emb_tensor    = torch.from_numpy(emb_array).float()
            image_embedding = emb_tensor 
        else:
            if image_path in self.image_embedding_cache:
                image_embedding = self.image_embedding_cache[image_path]
            else:
                image_embedding = self.image_text_inference.image_inference_engine.get_projected_global_embedding(image_path)
                if not demo:
                    self.image_embedding_cache[image_path] = image_embedding

        # Default get_similarity_score_from_raw_data would load the image every time. Instead we only load once.
        for desc in descriptors:
            if diagnosis:
                #prompt = f'evidence of {desc}'
                prompt = f'{desc}'
            else:
                prompt = f'There are {desc}'
            score = self.get_similarity_score_from_raw_data(image_embedding, prompt)
            if do_negative_prompting:
                if diagnosis:
                    #neg_prompt = f'no evidence of {desc}'
                    # WARNING: The test for diagnosis was done without the "no" prefix.
                    neg_prompt = f'no {desc}'
                else:
                    neg_prompt = f'There are no {desc}'
                neg_score = self.get_similarity_score_from_raw_data(image_embedding, neg_prompt)
            pos_prob = cos_sim_to_prob(score)
            if do_negative_prompting:
                #pos_prob, neg_prob = torch.softmax((torch.tensor([score, neg_score]) ), dim=0)
                # FIXME: Default temperature is 0.5
                pos_prob, neg_prob = torch.softmax((torch.tensor([score, neg_score]) / 0.5), dim=0)
                negative_probs[desc] = neg_prob
            probs[desc] = pos_prob

        return probs, negative_probs, image_embedding
    
    def get_prototype_probs(
            self,
            dicom_id, 
            image_path: Path, 
            disease_descriptors, 
            do_negative_prompting=True, 
            demo=False, 
            use_cache_file=False, 
            cached_image_embeddings_df=None,
            threshold_based=False,
            similarity_mode='max',):
        probs = {}
        negative_probs = {}
        if use_cache_file:
            # 1. Get the cached row as a Series
            row = cached_image_embeddings_df.loc[dicom_id]
            if row.empty:
                raise KeyError(f"No entry for dicom_id={dicom_id}")
            # 2. Convert to NumPy and then to tensor
            emb_array     = row.to_numpy()         
            emb_tensor    = torch.from_numpy(emb_array).float()
            image_embedding = emb_tensor 
        else:
            if image_path in self.image_embedding_cache:
                image_embedding = self.image_embedding_cache[image_path]
            else:
                image_embedding = self.image_text_inference.image_inference_engine.get_projected_global_embedding(image_path)
                if not demo:
                    self.image_embedding_cache[image_path] = image_embedding

        for disease, descs in disease_descriptors[0].items():
            score = self.get_similarity_score_from_raw_data(image_embedding, descs, similarity_mode=similarity_mode)
            pos_prob = cos_sim_to_prob(score)
            if do_negative_prompting:
                no_disease = f"No {disease}"
                neg_score = self.get_similarity_score_from_raw_data(
                    image_embedding, disease_descriptors[1][no_disease], similarity_mode=similarity_mode)
                pos_prob, neg_prob = torch.softmax((torch.tensor([score, neg_score])/0.5), dim=0)
                if not threshold_based:
                    negative_probs[disease] = neg_prob
                else:
                    # use the subtraction normalized to [0, 1] as the probability
                    probs[disease] = (pos_prob - neg_prob + 1) /2
            if not threshold_based:
                probs[disease] = pos_prob
        return probs, negative_probs, image_embedding

    def get_all_descriptors(self, disease_descriptors):
        all_descriptors = set()
        for disease, descs in disease_descriptors.items():
            all_descriptors.update([f"{desc} indicating {disease}" for desc in descs])
        all_descriptors = sorted(all_descriptors)
        return all_descriptors

    def get_all_descriptors_only_disease(self, disease_descriptors):
        all_descriptors = set()
        for disease, descs in disease_descriptors.items():
            all_descriptors.update([f"{desc}" for desc in descs])
        all_descriptors = sorted(all_descriptors)
        return all_descriptors

    def get_diseases_probs(self, disease_descriptors, pos_probs, negative_probs, prior_probs=None, do_negative_prompting=True, diagnosis=False):
        disease_probs = {}
        disease_neg_probs = {}
        for disease, descriptors in disease_descriptors.items():
            if diagnosis:
                disease_probs[disease] = pos_probs[descriptors[0]]
                if do_negative_prompting:
                    disease_neg_probs[disease] = negative_probs[descriptors[0]]
            else:
                desc_log_probs = []
                desc_neg_log_probs = []
                for desc in descriptors:
                    desc = f"{desc} indicating {disease}"
                    desc_log_probs.append(prob_to_log_prob(pos_probs[desc]))
                    if do_negative_prompting:
                        desc_neg_log_probs.append(prob_to_log_prob(negative_probs[desc]))
                disease_log_prob = sum(sorted(desc_log_probs, reverse=True)) / len(desc_log_probs)
                if do_negative_prompting:
                    disease_neg_log_prob = sum(desc_neg_log_probs) / len(desc_neg_log_probs)
                disease_probs[disease] = log_prob_to_prob(disease_log_prob)
                if do_negative_prompting:
                    disease_neg_probs[disease] = log_prob_to_prob(disease_neg_log_prob)

        return disease_probs, disease_neg_probs

    # Threshold Based
    def get_predictions(self, disease_descriptors, threshold, disease_probs, keys):
        predicted_diseases = torch.zeros(len(keys), dtype=torch.int)
        prob_vector = torch.zeros(len(keys), dtype=torch.float)  # num of diseases
        for disease in disease_descriptors:
            if disease == 'No Finding':
                continue
            # Find the index of the disease in keys.
            disease_idx = keys.index(disease)
            prob_vector[disease_idx] = disease_probs[disease]
            if disease_probs[disease] > threshold:
                predicted_diseases[disease_idx] = 1  # Predicted as positive.
            else:
                predicted_diseases[disease_idx] = 0  # Predicted as negative.
        if "No Findings" in keys:
            no_findings_idx = keys.index("No Findings")
            # The rule here is:
            # If no other disease is predicted as positive, set "No Findings" to 1.
            # Otherwise, set "No Findings" to be the complement of the maximum probability
            # among the other diseases.
            if torch.sum(predicted_diseases) == 0:
                predicted_diseases[no_findings_idx] = 1
            else:
                predicted_diseases[no_findings_idx] = 0
            # For the probability vector, we follow your previous logic.
            prob_vector[no_findings_idx] = 1.0 - max(prob_vector)

        return predicted_diseases, prob_vector

    # Negative vs Positive Prompting
    def get_predictions_bin_prompting(self, disease_descriptors, disease_probs, negative_disease_probs, keys):
        # Create a probability vector of length equal to the number of diseases.
        # We also create a binary prediction vector (0 = negative, 1 = positive).
        prob_vector = torch.zeros(len(keys), dtype=torch.float)
        neg_prob_vector = torch.zeros(len(keys), dtype=torch.float)
        predicted_vector = torch.zeros(len(keys), dtype=torch.int)  # binary predictions

        # Loop over all disease descriptors provided by the model.
        for disease in disease_descriptors:
            # Skip processing for 'No Finding'
            if disease == 'No Finding':
                continue
            # Get positive and negative scores for this disease.
            pos_score = disease_probs[disease]
            neg_score = negative_disease_probs[disease]
            # Find the index of the disease in keys.
            disease_idx = keys.index(disease)
            # Record the positive score into our probability vector.
            prob_vector[disease_idx] = pos_score
            neg_prob_vector[disease_idx] = neg_score
            # Make the prediction based on comparing positive and negative scores.
            if pos_score > neg_score:
                predicted_vector[disease_idx] = 1  # Predicted as positive.
            else:
                predicted_vector[disease_idx] = 0  # Predicted as negative.

        # Handle the 'No Findings' case.
        # We assume that 'No Findings' is included in keys.
        if "No Findings" in keys:
            no_findings_idx = keys.index("No Findings")
            # The rule here is:
            # If no other disease is predicted as positive, set "No Findings" to 1.
            # Otherwise, set "No Findings" to be the complement of the maximum probability
            # among the other diseases.
            if torch.sum(predicted_vector) == 0:
                predicted_vector[no_findings_idx] = 1
            else:
                predicted_vector[no_findings_idx] = 0
            # For the probability vector, we follow your previous logic.
            prob_vector[no_findings_idx] = 1.0 - max(prob_vector)
            neg_prob_vector[no_findings_idx] = max(prob_vector) 

        # Return the binary vector (predicted_vector) which is comparable with the labels,
        # and also the prob_vector if you need it for further analysis.
        return predicted_vector, prob_vector, neg_prob_vector
