import os
import librosa
import pandas as pd
import numpy as np
import re
from tabulate import tabulate
from itertools import combinations
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

#custom
from functions.func_dataset_management import create_dataframe

#--------------------------funzioni per il ml-----------------------------------

# Funzione per estrarre MFCCs e delta MFCCs da un file audio
def extract_features(file_path, window_size=2048, hop_length=512):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    #delta del delta
    # spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_entropy = librosa.feature.spectral_flatness(y=y)
    # spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # zcr = librosa.feature.zero_crossing_rate(y)
    # rms = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)
    # mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    features = np.vstack([mfccs, delta_mfccs, spectral_entropy 
                          # spectral_centroid, spectral_entropy, spectral_flux, rolloff, zcr, rms, mel_spectrogram, tempogram #
                          ])
    return features


def calculate_average_ratings(df_excel, excel_path, audio_folder):
    # Calcola la media dei voti per ciascun suono
    colonne_pazienti = df_excel.columns[1:]
    df_excel['Media_Voti'] = df_excel[colonne_pazienti].mean(axis=1)
    df_media_voti = df_excel[['File Audio', 'Media_Voti']]
    return create_dataframe(excel_path, audio_folder, df_media_voti)

def prepare_data(df, audio_folder):    
    df['features'] = df['file'].apply(lambda x: extract_features(os.path.join(audio_folder, x)))

    if not df.empty:
        # Preparazione dei dati
        X = np.vstack(df['features'].apply(lambda x: x.flatten()))
        y = df['label']
        scaler = StandardScaler()
        F = scaler.fit_transform(X)

        # Crea classi basate su intervalli di valori
        bins = [0, 2, 4, 6, 8, 10]
        labels = ['disturbo molto debole', 'disturbo debole', 'disturbo medio', 'disturbo forte', 'disturbo molto forte']

        df['label_discrete'] = pd.cut(df['label'], bins=bins, labels=labels, include_lowest=True)
        X_discrete = np.vstack(df['features'].apply(lambda x: x.flatten()))
        y_discrete = df['label_discrete']
        z = df['label']

        # Divisione dei dati in set di addestramento e test
        X_train, X_test, y_train, y_test = train_test_split(X_discrete, y_discrete, test_size=0.2, random_state=42)
        F_train, F_test, z_train, z_test = train_test_split(F, z, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test
    else:
        return None

def linear_regression(X_train, X_test, z_train, z_test, F_train, F_test):
    print("<------------------------Linear regression------------------------>")
    # Linear Regression senza normalizzazione
    linear_model = LinearRegression()
    linear_model.fit(X_train, z_train)
    linear_pred = linear_model.predict(X_test)
    mse_linear = mean_squared_error(z_test, linear_pred)
    mae_linear = mean_absolute_error(z_test, linear_pred)
    print(f'Regressione Lineare MSE: {mse_linear}')
    print(f'Regressione Lineare MAE: {mae_linear}')

    # Linear Regression con normalizzazione
    linear_modelf = LinearRegression()
    linear_modelf.fit(F_train, z_train)
    linearf_pred = linear_modelf.predict(F_test)
    mse_linearf = mean_squared_error(z_test, linearf_pred)
    mae_linearf = mean_absolute_error(z_test, linearf_pred)
    print(f'Regressione Lineare MSE w/normalization: {mse_linearf}')
    print(f'Regressione Lineare MAE w/normalization: {mae_linearf}')

    # Aggiungi qui ulteriori visualizzazioni o analisi necessarie

    return linear_pred, linearf_pred

def knn_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test):
    print("<------------------------KNN------------------------>")
    n_neighbors_values = [3, 5, 7, 9, 11]
    for n_neighbors in n_neighbors_values:
        # Addestramento del modello KNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        print(f'KNN (n_neighbors={n_neighbors}) Accuracy: {knn_accuracy}')

        # Addestramento del modello KNN normalizzato
        knnf = KNeighborsClassifier(n_neighbors=n_neighbors)
        knnf.fit(F_train, y_train)
        knnf_pred = knnf.predict(F_test)
        knnf_accuracy = accuracy_score(y_test, knnf_pred)
        print(f'KNN (n_neighbors={n_neighbors}) Accuracy w/normalization: {knnf_accuracy}')

    # Esecuzione della cross-validation per capire il numero di vicini ottimale
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Stampa dei risultati della cross-validation
    print()
    print()
    print("<------------------------Knn cross-validation------------------------>")
    print("Knn cross-validation:")
    print("Miglior parametro n_neighbors:", grid_search.best_params_)
    print("Migliore accuratezza durante la cross-validation:", grid_search.best_score_)

    # Addestramento del modello KNN con il miglior parametro trovato
    best_knn = grid_search.best_estimator_
    best_knn.fit(X_train, y_train)

    # Valutazione del modello sul set di test
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuratezza sul set di test:", accuracy)

    # Aggiungi qui ulteriori visualizzazioni o analisi necessarie

    return knn_pred, knnf_pred, y_pred, knn_accuracy, knnf_accuracy

def decision_tree_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test):
    print("<------------------------Decision Tree------------------------>")
    # Addestramento del modello Decision Tree
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    tree_accuracy = accuracy_score(y_test, tree_pred)
    print(f'Decision Tree Accuracy: {tree_accuracy}')

    # Addestramento del modello Decision Tree con normalizzazione
    treef = DecisionTreeClassifier()
    treef.fit(F_train, y_train)
    treef_pred = treef.predict(F_test)
    treef_accuracy = accuracy_score(y_test, treef_pred)
    print(f'Decision Tree Accuracy w/normalization: {treef_accuracy}')

    # Aggiungi qui ulteriori visualizzazioni o analisi necessarie

    return tree_pred, treef_pred, tree_accuracy, treef_accuracy

def svm_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test):
    print("<------------------------Support Vector Machine (SVM) standard------------------------>")
    # Addestramento del modello Support Vector Machine (SVM)
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f'SVM Accuracy: {svm_accuracy}')

    # Addestramento del modello Support Vector Machine con normalizzazione (SVM)
    svmf = SVC()
    svmf.fit(F_train, y_train)
    svmf_pred = svmf.predict(F_test)
    svmf_accuracy = accuracy_score(y_test, svmf_pred)
    print(f'SVM Accuracy w/normalization: {svmf_accuracy}')

    # Aggiungi qui ulteriori visualizzazioni o analisi necessarie

    print("<------------------------Support Vector Machine (SVM) varianti------------------------>")

     
    # Addestramento del modello Support Vector Machine (SVM) con kernel lineare senza normalizzazione
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    svm_linear_pred = svm_linear.predict(X_test)
    svm_linear_accuracy = accuracy_score(y_test, svm_linear_pred)
    print(f'SVM con kernel lineare senza normalizzazione Accuracy: {svm_linear_accuracy}')

    # Addestramento del modello Support Vector Machine (SVM) con kernel lineare e normalizzazione
    svmf_linear = SVC(kernel='linear')
    svmf_linear.fit(F_train, y_train)
    svmf_linear_pred = svmf_linear.predict(F_test)
    svmf_linear_accuracy = accuracy_score(y_test, svmf_linear_pred)
    print(f'SVM con kernel lineare e normalizzazione Accuracy: {svmf_linear_accuracy}')
    print()

    # Prova diversi gradi del kernel polinomiale
    poly_degrees = [2, 3, 4]  # Puoi aggiungere altri valori se desideri testare ulteriori gradi
    for degree in poly_degrees:
        # Addestramento del modello Support Vector Machine (SVM) con kernel polinomiale senza normalizzazione
        svm_poly = SVC(kernel='poly', degree=degree)
        svm_poly.fit(X_train, y_train)
        svm_poly_pred = svm_poly.predict(X_test)
        svm_poly_accuracy = accuracy_score(y_test, svm_poly_pred)
        print(f'SVM con kernel polinomiale (grado={degree}) senza normalizzazione Accuracy: {svm_poly_accuracy}')

        # Addestramento del modello Support Vector Machine (SVM) con kernel polinomiale e normalizzazione
        svmf_poly = SVC(kernel='poly', degree=degree)
        svmf_poly.fit(F_train, y_train)
        svmf_poly_pred = svmf_poly.predict(F_test)
        svmf_poly_accuracy = accuracy_score(y_test, svmf_poly_pred)
        print(f'SVM con kernel polinomiale (grado={degree}) e normalizzazione Accuracy: {svmf_poly_accuracy}')
        print()

    # Prova kernel gaussiano (RBF) senza normalizzazione
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    svm_rbf_pred = svm_rbf.predict(X_test)
    svm_rbf_accuracy = accuracy_score(y_test, svm_rbf_pred)
    print(f'SVM con kernel gaussiano senza normalizzazione Accuracy: {svm_rbf_accuracy}')

    # Prova kernel gaussiano (RBF) con normalizzazione
    svmf_rbf = SVC(kernel='rbf')
    svmf_rbf.fit(F_train, y_train)
    svmf_rbf_pred = svmf_rbf.predict(F_test)
    svmf_rbf_accuracy = accuracy_score(y_test, svmf_rbf_pred)
    print(f'SVM con kernel gaussiano e normalizzazione Accuracy: {svmf_rbf_accuracy}')
    print()

    return svm_pred, svmf_pred, svm_accuracy, svmf_accuracy, svm_linear_pred, svmf_linear_pred, svm_linear_accuracy, svmf_linear_accuracy, svm_poly_pred, svmf_poly_pred, svm_poly_accuracy, svmf_poly_accuracy, svm_rbf_pred, svmf_rbf_pred, svm_rbf_accuracy, svmf_rbf_accuracy

