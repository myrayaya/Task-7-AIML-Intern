# Task-7 : SVM Classification on Breast Cancer Dataset

## Overview
This repository consists code for binary classification of breast cancer (benign vs malignant) using Support Vector Machine (SVM) models with both Linear and RBF Kernal.

## Dataset
- Breast Cancer Dataset from Kaggle.com

## Steps
1. **Data Loading**: Loading and Preprocessing the dataset
2. **Model Training**:
   - Linear Kernel SVM
   - RBF Kernel SVM
3. **Hyperparamter Tuning**: Used GridSearchCV to find the best C and gamma for RBF Kernel
4. **Cross-Validation**: Evaluates model robustness using 5-fold CV
5. **Visualization** : Plots decision boundaries for both kernels using a 2D synthetic dataset

## Output
<img width="1020" height="908" alt="image" src="https://github.com/user-attachments/assets/92fc398b-0e16-489a-8cc1-3eb384ff6e9f" />
<img width="767" height="821" alt="image" src="https://github.com/user-attachments/assets/57353cce-9ffb-41dc-a021-c193d526c527" />
<img width="692" height="551" alt="image" src="https://github.com/user-attachments/assets/1ce4fee9-6d64-4125-9b01-d4fab2666c6d" />
<img width="703" height="547" alt="image" src="https://github.com/user-attachments/assets/5c2007d5-10e5-4d5a-a43a-4efaf11eee69" />

## Author
- Myra Chauhan
