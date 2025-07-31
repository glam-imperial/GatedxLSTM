# GatedxLSTM: A Multimodal Affective Computing Approach for Emotion Recognition in Conversations

## 📘 Introduction

This repository provides training and evaluation code for the paper 

**GatedxLSTM: A Multimodal Affective Computing Approach for Emotion Recognition in Conversations**. [[Paper Link]](https://arxiv.org/abs/2503.20919)


Key contributions of GatedxLSTM include:
- **CLAP-based Cross-modal Alignment**: Incorporating Contrastive Language-Audio Pretraining for improved speech-text alignment.
- **Gated Modality Fusion**: A gating mechanism to emphasise emotionally salient utterances.
- **Dialogical Emotion Decoder (DED)**: Captures context-aware emotional transitions over conversation turns.
---

## 🚀 How to Run

### Step 1: Download the Dataset
Download the [IEMOCAP dataset](https://sail.usc.edu/iemocap/) and extract it to your local directory.

Install required dependency packages.

### Step 2: Data Preprocessing
Update the dataset path in `./data/preprocess.py`, then run:
```
python preprocess.py
```

### Step 3: Training and Inference
To train and evaluate the model, run:
```
python ./Dialogical-Emotion-Decoding/main.py
```

## 📄 Citation
This work has been accepted at ACII 2025.

Welcome to cite our paper:
```
@article{li2025gatedxlstm,
  title={GatedxLSTM: A Multimodal Affective Computing Approach for Emotion Recognition in Conversations},
  author={Li, Yupei and Sun, Qiyang and Murthy, Sunil Munthumoduku Krishna and Alturki, Emran and Schuller, Bj{\"o}rn W},
  journal={arXiv preprint arXiv:2503.20919},
  year={2025}
}
```