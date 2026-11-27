# Xplainer on MIMIC-Diff-VQA (thesis fork)

This fork adapts [Xplainer](https://github.com/ChantalMP/Xplainer) (Pellegrini et al., MICCAI 2023) to detect diseases in MIMIC-Diff-VQA chest X-rays without task-specific training. The predictions become the text hints in my M.Sc. thesis, **[Context-Aware Difference VQA for Chest X-rays](https://github.com/sebtum/context-aware-diffvqa)**.

The method and the original code are the Xplainer authors' work. Their [README](#xplainer-from-x-ray-observations-to-explainable-zero-shot-diagnosis) follows below.

## What I changed

| Area | Change | Code |
|---|---|---|
| Dataset | MIMIC dataset class that reads images straight from tar archives | `mimic_dataset.py` |
| Preprocessing | Build the image list and labels, index the archives, resize so the shorter side is 512 px | `map_dicom_id_to_image_path.py`, `create_xplainer_dataset.py`, `build_mimic_tar_index.py`, `preprocess_mimic.py` |
| Descriptors | Sets for MIMIC: observation descriptors, disease names only, and short positive and negative phrases for prototyping | `descriptors.py` |
| Prototyping | Aggregate raw similarities per disease (max by default) instead of a joint probability | `model.py` |
| Speed and logging | Image embedding cache, per-disease precision, recall, F1, average precision and AUROC | `inference.py` |

## Result

The default Xplainer prompting won on 5 of 8 diseases. Average precision (thesis Table 5.3):

| Disease | Xplainer default | Diagnosis-name binary | "Evidence" + diagnosis binary | Prototype, ELIXR averaged | Prototype, ELIXR binary maxed | Prototype, ELIXR + diagnosis binary maxed | Prototype, prompt engineering |
|---|---|---|---|---|---|---|---|
| Atelectasis | **0.5601** | 0.3955 | 0.4706 | 0.4630 | 0.4704 | 0.4487 | 0.5438 |
| Cardiomegaly | 0.3885 | 0.3814 | 0.3791 | 0.3768 | **0.3895** | 0.3841 | 0.2965 |
| Consolidation | **0.1513** | 0.1291 | 0.0829 | 0.1032 | 0.1032 | 0.1190 | 0.0935 |
| Edema | **0.5185** | 0.4525 | 0.4858 | 0.4791 | 0.4600 | 0.4550 | 0.4127 |
| Lung Opacity | 0.4542 | **0.4871** | 0.3047 | 0.4679 | 0.4645 | 0.4688 | 0.3284 |
| Pleural Effusion | **0.7814** | 0.7666 | 0.5354 | 0.6985 | 0.6941 | 0.7351 | 0.6988 |
| Pneumonia | **0.2322** | 0.2241 | 0.1445 | 0.2217 | 0.2223 | 0.2174 | 0.1720 |
| Pneumothorax | 0.1083 | 0.1225 | 0.1140 | 0.1134 | 0.1134 | 0.1181 | **0.3172** |

[ELIXR](https://arxiv.org/abs/2308.01317) is the source of one of the descriptor sets.

Only three diseases were precise enough to produce hints (thesis Table 5.12):

| Disease | Threshold | Precision | Recall | F0.5 |
|---|---|---|---|---|
| Pleural Effusion | 0.650 | 0.792 | 0.586 | 0.740 |
| Atelectasis | 0.570 | 0.552 | 0.632 | 0.567 |
| Edema | 0.700 | 0.560 | 0.424 | 0.526 |

## Run it on MIMIC

This is research code. Paths are hard-coded to the thesis cluster, so edit them first. You need credentialed PhysioNet access to [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and [MIMIC-Diff-VQA](https://physionet.org/content/medical-diff-vqa/1.0.0/). These steps were read from the code and not re-run for this page.

```bash
git clone https://github.com/sebtum/Xplainer
cd Xplainer
conda create -n xplainer_env python=3.7
conda activate xplainer_env
pip install hi-ml-multimodal==0.1.2
pip install -r requirements.txt
pip install tqdm matplotlib   # also imported by the scripts
```

Prepare the data in this order. As committed, the switches are set for the extra reference images, so change them for the main image set.

| Step | Script | Switch to check |
|---|---|---|
| 1. Map image IDs to archive paths | `python map_dicom_id_to_image_path.py` | `plural` in `main()`; with `False`, also read the `question_type` column |
| 2. Attach disease labels (needs `compact_context_w_ref_id.csv`, which no script here creates) | `python create_xplainer_dataset.py` | `ref` in `main()` |
| 3. Add archive offsets | `python build_mimic_tar_index.py` | `ref_flag`, with `post_process = False` |
| 4. Resize images | `python preprocess_mimic.py` | single or parallel conversion at the bottom of the file |
| 5. Index the resized images | `python build_mimic_tar_index.py` | `single_process`, `post_process` |

Run inference. Use one output folder per strategy, because file names repeat.

```bash
mkdir -p results/default results/diagnosis results/prototyping results/ref
python inference.py --part 0 --output_folder results/default
python inference.py --part 0 --output_folder results/diagnosis --diagnosis
python inference.py --part 0 --output_folder results/prototyping --prototyping
python inference.py --ref --output_folder results/ref
```

| Detail | In the committed code |
|---|---|
| `--threshold_based` | Predict positive above 0.5 instead of comparing positive and negative scores |
| Prototyping set | ELIXR-style phrases plus the disease name, max pooling |
| Other table columns | The "evidence" prompts, mean pooling and prompt-engineering descriptors need code edits |
| Device | `model.py` fixes it to CPU |

---

# Xplainer: From X-Ray Observations to Explainable Zero-Shot Diagnosis
**Online Demo:** [![](https://img.shields.io/badge/🤗-Open_Gradio_Demo-blue)](https://huggingface.co/spaces/CAMP-ViL/Xplainer)
**Paper**: [![](https://img.shields.io/badge/Springer-978_3_031_43904_9-blue)](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_41) [![](https://img.shields.io/badge/Arxiv-2303.13391-blue)](https://arxiv.org/pdf/2303.13391.pdf)


This is the official repository for the paper "Xplainer: From X-Ray Observations to Explainable Zero-Shot Diagnosis", which was accepted for publication at MICCAI 2023.

<img src="figures/model_overview.png" alt="pipeline" width="100%"/>

We propose a new way of explainability for zero-shot diagnosis prediction in the clinical domain. Instead of directly predicting a diagnosis, we prompt the model to classify the existence of descriptive observations, which a radiologist would look for on an X-Ray scan, and use the descriptor probabilities to estimate the likelihood of a diagnosis, making our model explainable by design. For this we leverage BioVil, a pretrained CLIP model for X-rays and apply contrastive observation-based prompting. We evaluate Xplainer on two chest X-ray
datasets, CheXpert and ChestX-ray14, and demonstrate its effectiveness
in improving the performance and explainability of zero-shot diagnosis.
**Authors**: [Chantal Pellegrini][cp], [Matthias Keicher][mk], [Ege Özsoy][eo], [Petra Jiraskova][pj], Rickmer Braren, [Nassir Navab][nn]

[cp]:https://www.cs.cit.tum.de/camp/members/chantal-pellegrini/
[eo]:https://www.cs.cit.tum.de/camp/members/ege-oezsoy/
[mk]:https://www.cs.cit.tum.de/camp/members/matthias-keicher/
[pj]:https://campus.tum.de/tumonline/ee/ui/ca2/app/desktop/#/pl/ui/$ctx/visitenkarte.show_vcard?$ctx=design=ca2;header=max;lang=de&pPersonenGruppe=3&pPersonenId=46F3A857F258DEE6
[nn]:https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/

```
@article{pellegrini2023xplainer,
  title={Xplainer: From X-Ray Observations to Explainable Zero-Shot Diagnosis},
  author={Pellegrini, Chantal and Keicher, Matthias and {\"O}zsoy, Ege and Jiraskova, Petra and Braren, Rickmer and Navab, Nassir},
  journal={arXiv preprint arXiv:2303.13391},
  year={2023}
}
```

### Setup:
1. Clone this repository
   ```
   git clone https://github.com/ChantalMP/Xplainer
   ```
2. Install requirements:
   
   - use Python 3.7
   - install requirements:
   ```
   conda create -n xplainer_env python=3.7
   conda activate xplainer_env
   pip install hi-ml-multimodal==0.1.2
   pip install -r requirements.txt
   ```
   
3. Download data
   
   CheXpert:
   - download data from https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c
   - copy 'test_labels.csv' and the image folder 'test' into 'data/chexpert'
  
   ChestXRay14:
   - download data from 'https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345'
   - use the script 'preprocess_chestxray14' to download the data
   - copy 'images', 'test_list.txt' and 'Data_Entry_2017_v2020.csv' into 'data/chestxray14'
   - run
     ```
      python -m preprocesss_chestxray14
     ```

### Reproduce our results:
run
```
python -m inference --dataset chexpert
```
or
```
python -m inference --dataset chestxray14
```

### Run demo locally:
run
```
python -m demo
```

### Intended Use
This model is intended to be used solely for (I) future research on visual-language processing and (II) reproducibility of the experimental results reported in the reference paper. 
