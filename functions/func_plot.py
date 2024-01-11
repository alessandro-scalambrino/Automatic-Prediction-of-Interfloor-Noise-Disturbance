import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results(z_test, linear_pred, linearf_pred, y_test, knn_pred, knn_accuracy, knnf_pred, knnf_accuracy,
                 tree_pred, tree_accuracy, treef_pred, treef_accuracy, svm_pred, svm_accuracy, svmf_pred,
                 svmf_accuracy, svm_linear_pred, svmf_linear_pred, svm_linear_accuracy, svmf_linear_accuracy,
                 svm_poly_pred, svmf_poly_pred, svm_poly_accuracy, svmf_poly_accuracy, svm_rbf_pred, svmf_rbf_pred,
                 svm_rbf_accuracy, svmf_rbf_accuracy):

    # Grafici per regressione lineare
    plt.scatter(z_test, linear_pred, color='blue', label='Valori Attuali vs Predetti (Senza Normalizzazione)')
    plt.title('Grafico a Dispersione - Linear Regression (Senza Normalizzazione)')
    plt.xlabel('Valori Attuali')
    plt.ylabel('Valori Predetti')
    plt.legend()
    plt.show()

    plt.scatter(z_test, linearf_pred, color='green', label='Valori Attuali vs Predetti (Con Normalizzazione)')
    plt.title('Grafico a Dispersione - Linear Regression (Con Normalizzazione)')
    plt.xlabel('Valori Attuali')
    plt.ylabel('Valori Predetti')
    plt.legend()

    # Calcolo matrici di confusione
    
    cm_knn = confusion_matrix(y_test, knn_pred)
    cm_tree = confusion_matrix(y_test, tree_pred)
    cm_svm = confusion_matrix(y_test, svm_pred)

    # Visualizzazione matrici di confusione
    categories = ['molto debole', 'debole', 'medio', 'forte', 'molto forte']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    

    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], xticklabels=categories,
                yticklabels=categories)
    axes[0].set_title('KNN Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1], xticklabels=categories,
                yticklabels=categories)
    axes[1].set_title('Decision Tree Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[2], xticklabels=categories,
                yticklabels=categories)
    axes[2].set_title('SVM Confusion Matrix')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')

    # Visualizzazione del diagramma
    plt.show()

    plt.figure()

    # Accuratezze dei modelli
    accuracies = [knn_accuracy, tree_accuracy, svm_accuracy, knnf_accuracy, treef_accuracy, svmf_accuracy,
                  svm_linear_accuracy, svmf_linear_accuracy, svm_poly_accuracy, svmf_poly_accuracy,
                  svm_rbf_accuracy, svmf_rbf_accuracy]

    # Nomi dei modelli
    models = ['KNN', 'Decision Tree', 'SVM', 'KNN w/normalization', 'Decision Tree w/normalization',
              'SVM w/normalization', 'SVM Linear', 'SVM Linear w/normalization', 'SVM Poly', 'SVM Poly w/normalization',
              'SVM RBF', 'SVM RBF w/normalization']

    # Creazione del diagramma a barre
    plt.bar(models, accuracies, color=['blue', 'green', 'red', 'blue', 'green', 'red', 'purple', 'orange', 'pink', 'brown', 'cyan', 'gray'])

    # Aggiunta di etichette e titolo
    plt.xlabel('Modelli')
    plt.ylabel('Accuratezza')
    plt.title('Accuratezza dei modelli di classificazione')

    # Visualizzazione del diagramma
    plt.show()

