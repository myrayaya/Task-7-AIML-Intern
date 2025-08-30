# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification

# importing the dataset
df = pd.read_csv('dataset/breast-cancer.csv')

# exploring the data
print(df.head())
print(df.info())

# inspecting and preprocessing
target_col = df.columns[0]
df = df.drop(target_col, axis = 1)
x = df.drop('diagnosis', axis = 1)
y = df['diagnosis'].map({'B' : 0, 'M' : 1})

# standardizing the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)

# svm (linear kernel)
svm_linear = SVC(kernel = 'linear', C = 1.0, random_state = 42)
svm_linear.fit(x_train, y_train)
y_pred_linear = svm_linear.predict(x_test)
print('\nLinear SVM')
print('Accuracy: ', accuracy_score(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# SVM (RBF kernel)
svm_rbf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale', random_state = 42)
svm_rbf.fit(x_train, y_train)
y_pred_rbf = svm_rbf.predict(x_test)
print('\nSVM RBF:-')
print('Accuracy: ', accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# hyperparameter tuning
param_grid = {'C' : [0.1, 1, 10], 'gamma' : ['scale', 0.1, 1], 'kernel' : ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv = 5)
grid.fit(x_train, y_train)
print('\nBest Parameters: ', grid.best_params_)
print('Best CV Score: ', grid.best_score_)

# cross-validation
best_svm = grid.best_estimator_
cv_scores = cross_val_score(best_svm, x_scaled, y, cv = 5)
print('\nCross-Validation Scores : ', cv_scores)
print('Mean CV Accuracy : ', np.mean(cv_scores))

# decision boundary (2D synthetic demo)
x_vis, y_vis = make_classification(n_samples = 200, n_features = 2, n_redundant = 0, n_informative = 2, random_state = 42)
x_vis = StandardScaler().fit_transform(x_vis)
x_vis_train, x_vis_test, y_vis_train, y_vis_test = train_test_split(x_vis, y_vis, test_size = 0.2, random_state = 42)
svm_lin_vis = SVC(kernel = 'linear', C = 1).fit(x_vis_train, y_vis_train)
svm_rbf_vis = SVC(kernel = 'rbf', C = 1, gamma = 0.5).fit(x_vis_train, y_vis_train)

def plot_dec_bound(clf, x, y, title):
    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha = 0.3)
    plt.scatter(x[:, 0], x[:, 1], c = y, edgecolors = 'k')
    plt.title(title)
    plt.show()

plot_dec_bound(svm_lin_vis, x_vis, y_vis, 'SVM with Linear Kernel')
plot_dec_bound(svm_rbf_vis, x_vis, y_vis, 'SVM with RBF Kernel')

