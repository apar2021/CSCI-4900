# CDANET: Convolutional Dual-Attention Network for Brain Tumor Classification

## Important Links
- [CDANet Presentation](https://docs.google.com/presentation/d/113aG7IPbJ4JlwCnHFGENeLVvkC7v_OmkL6FyI-s4ygQ/edit?usp=sharing)
- [Dataset on Kaggle](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset/)

## Project Overview
CDANET (Convolutional Dual-Attention Network) is a deep learning model designed for accurate classification of brain MRI tumor images. It combines an EfficientNet backbone with attention mechanisms to achieve high-performance brain tumor classification into three categories: Meningioma, Glioma, and Pituitary tumors.

## Key Features
- **Advanced Architecture**: Utilizes EfficientNet-B4 as the backbone with custom attention modules (CBAM, Self-Attention, Dual-Path)
- **High Accuracy**: Achieves ~97% accuracy on the Figshare Brain Tumor Dataset
- **Multiple Model Variants**: Supports three model configurations:
  - `efficient_basic`: Base model with Self-Attention and SE blocks
  - `efficient_cbam`: Enhanced model with Convolutional Block Attention Module
  - `efficient_dual`: Dual path architecture for parallel feature extraction
- **Robust Training**: Implements advanced training techniques including:
  - MixUp augmentation
  - Label smoothing
  - OneCycle learning rate scheduling
  - Extensive data augmentation

## Dataset
The model is trained and evaluated on the Figshare Brain Tumor Dataset containing 3,064 T1-weighted contrast-enhanced MRI images of three types of brain tumors:
- Meningioma (Class 0)
- Glioma (Class 1)
- Pituitary (Class 2)

## Performance
The model achieves impressive results across 5-fold cross-validation:
- Average accuracy: 96.73%
- Average micro-average AUC: 0.9972
- Average macro-average AUC: 0.9971

## Requirements
```
torch>=2.5.0
torchvision>=0.20.0
opencv-python>=4.11.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
torchsummary>=1.5.0
kagglehub>=0.3.0
```

## Model Architecture
The architecture combines:
1. EfficientNet-B4 backbone
2. Attention mechanisms (CBAM/Self-Attention/Dual-Path)
3. Feature Pyramid Network (FPN) modules
4. Global Context Block
5. Advanced classifier head with regularization

## Usage
1. Install the required dependencies:
   ```bash
   pip install torch torchvision opencv-python numpy pandas matplotlib scikit-learn torchsummary kagglehub
   ```

2. Run the notebook to train and evaluate the model:
   ```python
   # The model configuration can be adjusted through CONFIG dictionary:
   CONFIG = {
       'batch_size': 16,
       'num_epochs': 15,
       'base_lr': 5e-4,
       'weight_decay': 2e-5,
       'scheduler': 'onecycle',
       'mixup_alpha': 0.2,
       'label_smoothing': 0.1,
       'dropout_rate': 0.4,
       'model_variant': 'efficient_cbam',  # Options: 'efficient_basic', 'efficient_cbam', 'efficient_dual'
       'num_classes': 3
   }
   ```

## Results Visualization
The training process generates comprehensive visualizations:
- ROC curves for each tumor type
- Confusion matrices
- Learning curves
- Class-wise performance metrics

## Citation
If you use this model in your research, please cite:
```
@INPROCEEDINGS{9897799,
  author={Kumar Dutta, Tapas and Ranjan Nayak, Deepak},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)}, 
  title={CDANet: Channel Split Dual Attention Based CNN for Brain Tumor Classification In Mr Images}, 
  year={2022},
  volume={},
  number={},
  pages={4208-4212},
  doi={10.1109/ICIP46576.2022.9897799}
}
