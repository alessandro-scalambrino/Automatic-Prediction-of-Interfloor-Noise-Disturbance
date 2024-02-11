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
                 svm_rbf_accuracy, svmf_rbf_accuracy, mse_linear, mae_linear, mse, mae, tree_reg_mse, tree_reg_mae, 
                 svr_linear_mse, svr_linear_mae, svr_poly_mse, svr_poly_mae, svr_rbf_mse, svr_rbf_mae):

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

    # Visualizzazione del diagramma per la classificazione
    plt.figure()

    accuracies_classification = [knn_accuracy, knnf_accuracy, tree_accuracy,  treef_accuracy, svm_accuracy,  svmf_accuracy,
                  svm_linear_accuracy, svmf_linear_accuracy, svm_poly_accuracy, svmf_poly_accuracy,
                  svm_rbf_accuracy, svmf_rbf_accuracy]

    models_classification = ['KNN', 'KNN w/n', 'Decision Tree', 'Decision Tree w/n', 'SVM', 
                  'SVM w/n', 'SVM Lin', 'SVM Lin w/n', 'SVM Poly', 'SVM Poly w/n',
                  'SVM RBF', 'SVM RBF w/n']

    # Definizione dei colori con diverse sfumature
    colors = ['dodgerblue', 'dodgerblue', 'skyblue', 'skyblue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
    alphas = [1.0, 0.7, 1.0, 0.7, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    for model, accuracy, color, alpha in zip(models_classification, accuracies_classification, colors, alphas):
        plt.bar(model, accuracy, color=color, alpha=alpha)

    plt.xlabel('Modelli')
    plt.ylabel('Accuratezza')
    plt.title('Accuratezza dei modelli di classificazione')

    # Rotazione delle etichette sull'asse x in diagonale
    plt.xticks(rotation=45, ha='right')

    plt.show()

    # Visualizzazione del diagramma per la regressione
    plt.figure()

    accuracies_regression = [mse_linear, mae_linear, mse, mae, tree_reg_mse, tree_reg_mae, svr_linear_mse, svr_linear_mae, svr_poly_mse, svr_poly_mae, svr_rbf_mse, svr_rbf_mae]
    models_regression = ['linear regression mse', 'linear regression mae', 'KNN mse', 'KNN mae','Decision Tree mse','Decision Tree mae', 'SVR lin mse', 'SVR lin mae', 'SVR poly mse', 'SVR poly mae', 'SVR gauss mse', 'SVR gauss mae']

    # Definizione dei colori con diverse sfumature
    colors = ['navy', 'navy', 'dodgerblue', 'dodgerblue', 'skyblue', 'skyblue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
    alphas = [1.0, 0.7, 1.0, 0.7, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    # Disegna le barre con colori e sfumature diverse
    for model, accuracy, color, alpha in zip(models_regression, accuracies_regression, colors, alphas):
        plt.bar(model, accuracy, color=color, alpha=alpha)

    plt.xlabel('Modelli')
    plt.ylabel('Errore')
    plt.title('Errore quadratico medio e assoluto dei modelli di regressione')

    # Rotazione delle etichette sull'asse x in diagonale
    plt.xticks(rotation=45, ha='right')

    plt.show()
