# Notebooks

This directory is for training notebooks and experimentation.

## Kaggle Notebook

The complete training process was conducted on **Kaggle** using their free GPU resources.

**Live Notebook**: [Brain Tumor Classification - Hybrid Deep Learning](https://www.kaggle.com/code/harisankera/brain-tumor-classification-hybrid-deep-learning)

### What's in the Kaggle Notebook:

- Data loading and preprocessing from Kaggle datasets
- Exploratory Data Analysis (EDA) with visualizations
- Hybrid ViT-L16 + Xception model architecture definition
- Model training with GPU acceleration (P100)
- Performance evaluation and metrics
- Confusion matrix and classification reports
- Model export for deployment

## Option: Download Notebook Locally

You can download the `.ipynb` file from Kaggle and add it here for offline reference:

1. Go to the [Kaggle notebook](https://www.kaggle.com/code/harisankera/brain-tumor-classification-hybrid-deep-learning)
2. Click "Copy & Edit" or "Download" 
3. Save the `.ipynb` file in this folder
4. Commit to this repository

## Running Locally

If you download the notebook:

```bash
jupyter notebook brain-tumor-classification.ipynb
```

**Note**: Training requires GPU (Tesla P100 or better) and ~3.5 hours.
