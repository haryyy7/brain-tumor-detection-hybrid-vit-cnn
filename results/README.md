# Results

This directory contains model training results, metrics, and visualizations.

## Contents

### Model Performance
- **Test Accuracy**: 96.4%
- **Training Accuracy**: 98.2%
- **Precision**: 96.8%
- **Recall**: 96.2%
- **F1-Score**: 96.5%

### Files
- `training_history.json` - Training metrics history
- `confusion_matrix.png` - Confusion matrix visualization
- `accuracy_plot.png` - Training/validation accuracy curves
- `loss_plot.png` - Training/validation loss curves

- ## How to Add Results from Kaggle

The visualizations (confusion matrix, accuracy/loss curves) are generated in the Kaggle notebook.

### Steps to Export Results:

1. **Run your [Kaggle notebook](https://www.kaggle.com/code/harisankera/brain-tumor-classification-hybrid-deep-learning)**
2. **Generate plots** in the notebook cells:
   - Training/Validation accuracy curve
   - Training/Validation loss curve  
   - Confusion matrix
   - Classification report

3. **Save plots** in Kaggle:
```python
import matplotlib.pyplot as plt

# Save accuracy plot
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')

# Save loss plot  
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')

# Save confusion matrix
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
```

4. **Download files** from Kaggle:
   - Click on "Output" tab in Kaggle
   - Download the generated PNG files

5. **Upload to this folder**:
   - Drag and drop the images into this `results/` directory
   - Commit with message: "Add training visualizations from Kaggle"

### Expected Files:

- `confusion_matrix.png` - Shows how well model predicts each class
- `accuracy_plot.png` - Training vs Validation accuracy over epochs
- `loss_plot.png` - Training vs Validation loss over epochs  
- `classification_report.txt` - Per-class precision, recall, F1-score
- `classification_report.txt` - Detailed classification metrics

## Model Summary
The Hybrid ViT-L16-Xception model achieves excellent performance on the brain tumor MRI classification task, with balanced precision and recall across all tumor classes (Glioma, Meningioma, Pituitary, No Tumor).
