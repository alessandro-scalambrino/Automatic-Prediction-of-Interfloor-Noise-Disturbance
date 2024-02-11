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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score


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
    #spectral_entropy = librosa.feature.spectral_flatness(y=y)
    # spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # zcr = librosa.feature.zero_crossing_rate(y)
    # rms = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)
    # mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    #onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    #tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    features = np.vstack([mfccs, delta_mfccs, #spectral_entropy 
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
        F = scaler.fit(X)
        F= scaler.transform(X)

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
        # X sono le feature, Y le label discrete, F le feature normalizzate, z le label continue
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
    r2_linear = linear_model.score(X_test, z_test)

    print(f'Regressione Lineare MSE: {mse_linear}')
    print(f'Regressione Lineare MAE: {mae_linear}')
    print(f'Regressione Lineare R^2: {r2_linear}')

    # Linear Regression con normalizzazione
    linear_modelf = LinearRegression()
    linear_modelf.fit(F_train, z_train)
    linearf_pred = linear_modelf.predict(F_test)
    mse_linearf = mean_squared_error(z_test, linearf_pred)
    mae_linearf = mean_absolute_error(z_test, linearf_pred)
    r2_linearf = linear_modelf.score(F_test, z_test)

    print(f'Regressione Lineare MSE w/normalization: {mse_linearf}')
    print(f'Regressione Lineare MAE w/normalization: {mae_linearf}')
    print(f'Regressione Lineare R^2 w/normalization: {r2_linearf}')

    return linear_pred, linearf_pred, mse_linear, mae_linear

def knn_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test):
    print("<------------------------KNN------------------------>")
    n_neighbors_values = [3, 5, 7, 9, 11]
    for n_neighbors in n_neighbors_values:
        # Classificazione: Addestramento del modello KNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        print(f'KNN Classification (n_neighbors={n_neighbors}) Accuracy: {knn_accuracy}')

        # Classificazione: Addestramento del modello KNN normalizzato
        knnf = KNeighborsClassifier(n_neighbors=n_neighbors)
        knnf.fit(F_train, y_train)
        knnf_pred = knnf.predict(F_test)
        knnf_accuracy = accuracy_score(y_test, knnf_pred)
        print(f'KNN Classification (n_neighbors={n_neighbors}) Accuracy w/normalization: {knnf_accuracy}')
        
        # Regressione: Addestramento del modello KNN 
        knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn_regressor.fit(X_train, z_train)  # Utilizza z_train come etichette continue per la regressione
        knn_pred_regression = knn_regressor.predict(X_test)
        mse_regression = mean_squared_error(z_test, knn_pred_regression)  # Calcola l'errore quadratico medio
        mae_regression = mean_absolute_error(z_test, knn_pred_regression)  # Calcola l'errore medio assoluto
        print(f'KNN Regression (n_neighbors={n_neighbors}) MSE: {mse_regression}')
        print(f'KNN Regression (n_neighbors={n_neighbors}) MAE: {mae_regression}')

        # Regressione: Addestramento del modello KNN normalizzato
        knnf_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
        knnf_regressor.fit(F_train, z_train)  # Utilizza z_train come etichette continue per la regressione
        knnf_pred_regression = knnf_regressor.predict(F_test)
        mse_regression_normalized = mean_squared_error(z_test, knnf_pred_regression)  # Calcola l'errore quadratico medio
        mae_regression_normalized = mean_absolute_error(z_test, knnf_pred_regression)  # Calcola l'errore medio assoluto
        print(f'KNN Regression (n_neighbors={n_neighbors}) MSE w/normalization: {mse_regression_normalized}')
        print(f'KNN Regression (n_neighbors={n_neighbors}) MAE w/normalization: {mae_regression_normalized}')

   # Esecuzione della cross-validation per capire il numero di vicini ottimale per compiti di classificazione e regressione
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    grid_search_regressor = GridSearchCV(knn_regressor, param_grid, cv=5, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'], refit=False)
    grid_search_regressor.fit(X_train, z_train) 

    # Stampa dei risultati della cross-validation
    print("<------------------------Knn cross-validation------------------------>")
    print("Knn classifier cross-validation:")
    print("Miglior parametro n_neighbors:", grid_search.best_params_)
    print("Migliore accuratezza durante la cross-validation:", grid_search.best_score_)
    print("Knn regressor cross-validation:")
    best_index_regressor = np.argmax(grid_search_regressor.cv_results_['mean_test_neg_mean_squared_error'])
    best_params_regressor = grid_search_regressor.cv_results_['params'][best_index_regressor]
    best_mse_regressor = -grid_search_regressor.cv_results_['mean_test_neg_mean_squared_error'][best_index_regressor]
    best_mae_regressor = -grid_search_regressor.cv_results_['mean_test_neg_mean_absolute_error'][best_index_regressor]
    print("Miglior parametro n_neighbors:", best_params_regressor)
    print("Miglior punteggio MSE durante la cross-validation:", best_mse_regressor)
    print("Miglior punteggio MAE durante la cross-validation:", best_mae_regressor)

    # Addestramento del modello KNN con il miglior parametro trovato
    best_knn = KNeighborsClassifier(**grid_search.best_params_)
    best_knn.fit(X_train, y_train)
    best_knn_regressor = KNeighborsRegressor(**best_params_regressor)
    best_knn_regressor.fit(X_train, z_train)  # Utilizza z_train come etichette continue per la regressione

    # Valutazione del modello sul set di test
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("<------------------------Knn Classification------------------------>")
    print("Accuratezza sul set di test nella classificazione:", accuracy)

    z_pred = best_knn_regressor.predict(X_test)
    mse = mean_squared_error(z_test, z_pred)  
    mae = mean_absolute_error(z_test, z_pred) 
    print("<------------------------Knn Regression------------------------>")
    print("MSE sul set di test:", mse)
    print("MAE sul set di test:", mae)

    return knn_pred, knnf_pred, y_pred, knn_accuracy, knnf_accuracy, mse, mae

def decision_tree_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test):
    print("<------------------------Decision Tree------------------------>")
    # Addestramento del modello Decision Tree
    best_tree_params = {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}

    tree = DecisionTreeClassifier(**best_tree_params)
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
    
    # Addestramento del modello Decision Tree per la regressione
    tree_reg = DecisionTreeRegressor(**best_tree_params)  
    tree_reg.fit(X_train, z_train)  # Utilizza z_train come etichette continue per la regressione
    tree_reg_pred = tree_reg.predict(X_test)
    tree_reg_mse = mean_squared_error(z_test, tree_reg_pred)
    tree_reg_mae = mean_absolute_error(z_test, tree_reg_pred)
    print(f'Decision Tree Regression MSE: {tree_reg_mse}')
    print(f'Decision Tree Regression MAE: {tree_reg_mae}')
    
    # Addestramento del modello Decision Tree per la regressione con normalizzazione
    treef_reg = DecisionTreeRegressor()
    treef_reg.fit(F_train, z_train)  # Utilizza z_train come etichette continue per la regressione
    treef_reg_pred = treef_reg.predict(F_test)
    treef_reg_mse = mean_squared_error(z_test, treef_reg_pred)
    treef_reg_mae = mean_absolute_error(z_test, treef_reg_pred)
    print(f'Decision Tree Regression MSE w/normalization: {treef_reg_mse}')
    print(f'Decision Tree Regression MAE w/normalization: {treef_reg_mae}')

    
    #CROSS VALIDATION PER TROVARE I PARAMETRI MIGLIORI
    # param_grid = {'max_depth': [None, 10, 20, 30],
              # 'min_samples_split': [2, 5, 10],
              # 'min_samples_leaf': [1, 2, 4]}
    # grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    # grid_search.fit(X_train, y_train)
    # best_tree = grid_search.best_estimator_
    # tree_pruned = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=3)
    # tree_pruned.fit(X_train, y_train)
    # print("Best Parameters from Grid Search:")
    # print(grid_search.best_params_)
    
    # Addestramento del modello Random Forest per la classificazione
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)
    forest_accuracy = accuracy_score(y_test, forest.predict(X_test))
    print(f'Random Forest Accuracy: {forest_accuracy}')
    
    # # Addestramento del modello Random Forest per la regressione
    # forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    # forest_reg.fit(X_train, z_train)  # Utilizza z_train come etichette continue per la regressione
    # forest_reg_pred = forest_reg.predict(X_test)
    # forest_reg_mse = mean_squared_error(z_test, forest_reg_pred)
    # forest_reg_mae = mean_absolute_error(z_test, forest_reg_pred)
    # print(f'Random Forest Regression MSE: {forest_reg_mse}')
    # print(f'Random Forest Regression MAE: {forest_reg_mae}')

    return tree_pred, treef_pred, tree_accuracy, treef_accuracy, tree_reg_mse, tree_reg_mae

def svm_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test):
    print("<------------------------Support Vector Machine (SVM) standard------------------------>")
    
    # svm = SVC()
    # svm_pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

    # # Definizione dei parametri da testare, utilizzando 'svm__' come prefisso per i parametri della SVM nel Pipeline
    # param_grid = {'svm__C': [0.1, 1, 10, 100],
                  # 'svm__gamma': [0.01, 0.1, 1, 10],
                  # 'svm__degree': [2, 3, 4]}

    # # Utilizzo di GridSearchCV per trovare i migliori iperparametri
    # grid_search = GridSearchCV(svm_pipe, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train, y_train)

    # # Stampare i risultati completi della ricerca
    # print("Grid Search Results:")
    # for params, mean_score, std_score in zip(grid_search.cv_results_['params'],
                                              # grid_search.cv_results_['mean_test_score'],
                                              # grid_search.cv_results_['std_test_score']):
        # print(f'Parameters: {params}, Mean Accuracy: {mean_score}, Standard Deviation: {std_score}')

    # # Valutazione del modello con i migliori iperparametri su un set di validazione
    # best_svm = grid_search.best_estimator_
    # best_svm_accuracy = accuracy_score(y_test, best_svm.predict(X_test))
    # print(f'Best SVM Accuracy on Validation Set: {best_svm_accuracy}')

    # # Stampa i migliori parametri della ricerca
    # print("Best Parameters from Grid Search:")
    # print(grid_search.best_params_)
    
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


    print("<------------------------Support Vector Machine (SVM) varianti------------------------>")

     
    # Addestramento del modello Support Vector Machine (SVM) con kernel lineare senza normalizzazione
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    svm_linear_pred = svm_linear.predict(X_test)
    svm_linear_accuracy = accuracy_score(y_test, svm_linear_pred)
    print(f'SVM con kernel lineare senza normalizzazione Accuracy: {svm_linear_accuracy}')
    
    # Addestramento del modello Support Vector Machine (SVM) per la regressione con kernel lineare senza normalizzazione
    svr_linear = SVR(kernel='linear')
    svr_linear.fit(X_train, z_train)  # Usa z_train come etichette continue per la regressione
    svr_linear_pred = svr_linear.predict(X_test)
    svr_linear_mse = mean_squared_error(z_test, svr_linear_pred)
    svr_linear_mae = mean_absolute_error(z_test, svr_linear_pred)
    print(f'SVM con kernel lineare per la regressione senza normalizzazione MSE: {svr_linear_mse}')
    print(f'SVM con kernel lineare per la regressione senza normalizzazione MAE: {svr_linear_mae}')


    # Addestramento del modello Support Vector Machine (SVM) con kernel lineare e normalizzazione
    svmf_linear = SVC(kernel='linear')
    svmf_linear.fit(F_train, y_train)
    svmf_linear_pred = svmf_linear.predict(F_test)
    svmf_linear_accuracy = accuracy_score(y_test, svmf_linear_pred)
    print(f'SVM con kernel lineare e normalizzazione Accuracy: {svmf_linear_accuracy}')
    print()
    
    # Addestramento del modello Support Vector Machine (SVM) per la regressione con kernel lineare e normalizzazione
    svrf_linear = SVR(kernel='linear')
    svrf_linear.fit(F_train, z_train)  # Usa z_train come etichette continue per la regressione
    svrf_linear_pred = svrf_linear.predict(F_test)
    svrf_linear_mse = mean_squared_error(z_test, svrf_linear_pred)
    svrf_linear_mae = mean_absolute_error(z_test, svrf_linear_pred)
    print(f'SVM con kernel lineare per la regressione con normalizzazione MSE: {svrf_linear_mse}')
    print(f'SVM con kernel lineare per la regressione con normalizzazione MAE: {svrf_linear_mae}')

    # Prova diversi gradi del kernel polinomiale
    poly_degrees = [2, 3, 4] 
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
        print(f'SVM con kernel polinomiale (grado={degree}) con normalizzazione Accuracy: {svmf_poly_accuracy}')
        print()
        
        # Addestramento del modello Support Vector Machine (SVM) per la regressione con kernel polinomiale senza normalizzazione
        svr_poly = SVR(kernel='poly', degree=degree)
        svr_poly.fit(X_train, z_train)  # Usa z_train come etichette continue per la regressione
        svr_poly_pred = svr_poly.predict(X_test)
        svr_poly_mse = mean_squared_error(z_test, svr_poly_pred)
        svr_poly_mae = mean_absolute_error(z_test, svr_poly_pred)
        print(f'SVM con kernel polinomiale (grado={degree}) per la regressione senza normalizzazione MSE: {svr_poly_mse}')
        print(f'SVM con kernel polinomiale (grado={degree}) per la regressione senza normalizzazione MAE: {svr_poly_mae}')

        # Addestramento del modello Support Vector Machine (SVM) per la regressione con kernel polinomiale e normalizzazione
        svrf_poly = SVR(kernel='poly', degree=degree)
        svrf_poly.fit(F_train, z_train)  # Usa z_train come etichette continue per la regressione
        svrf_poly_pred = svrf_poly.predict(F_test)
        svrf_poly_mse = mean_squared_error(z_test, svrf_poly_pred)
        svrf_poly_mae = mean_absolute_error(z_test, svrf_poly_pred)
        print(f'SVM con kernel polinomiale (grado={degree}) per la regressione con normalizzazione MSE: {svrf_poly_mse}')
        print(f'SVM con kernel polinomiale (grado={degree}) per la regressione con normalizzazione MAE: {svrf_poly_mae}')


    # Prova kernel gaussiano (RBF) senza normalizzazione
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    svm_rbf_pred = svm_rbf.predict(X_test)
    svm_rbf_accuracy = accuracy_score(y_test, svm_rbf_pred)
    print(f'SVM con kernel gaussiano senza normalizzazione Accuracy: {svm_rbf_accuracy}')
    
    # Addestramento del modello Support Vector Machine (SVM) per la regressione con kernel gaussiano (RBF) senza normalizzazione
    svr_rbf = SVR(kernel='rbf')
    svr_rbf.fit(X_train, z_train)  # Usa z_train come etichette continue per la regressione
    svr_rbf_pred = svr_rbf.predict(X_test)
    svr_rbf_mse = mean_squared_error(z_test, svr_rbf_pred)
    svr_rbf_mae = mean_absolute_error(z_test, svr_rbf_pred)
    print(f'SVM con kernel gaussiano (RBF) per la regressione senza normalizzazione MSE: {svr_rbf_mse}')
    print(f'SVM con kernel gaussiano (RBF) per la regressione senza normalizzazione MAE: {svr_rbf_mae}')

    # Prova kernel gaussiano (RBF) con normalizzazione
    svmf_rbf = SVC(kernel='rbf')
    svmf_rbf.fit(F_train, y_train)
    svmf_rbf_pred = svmf_rbf.predict(F_test)
    svmf_rbf_accuracy = accuracy_score(y_test, svmf_rbf_pred)
    print(f'SVM con kernel gaussiano con normalizzazione Accuracy: {svmf_rbf_accuracy}')
    print()
    
    # Addestramento del modello Support Vector Machine (SVM) per la regressione con kernel gaussiano (RBF) e normalizzazione
    svrf_rbf = SVR(kernel='rbf')
    svrf_rbf.fit(F_train, z_train)  # Usa z_train come etichette continue per la regressione
    svrf_rbf_pred = svrf_rbf.predict(F_test)
    svrf_rbf_mse = mean_squared_error(z_test, svrf_rbf_pred)
    svrf_rbf_mae = mean_absolute_error(z_test, svrf_rbf_pred)
    print(f'SVM con kernel gaussiano (RBF) per la regressione con normalizzazione MSE: {svrf_rbf_mse}')
    print(f'SVM con kernel gaussiano (RBF) per la regressione con normalizzazione MAE: {svrf_rbf_mae}')
    
    return svm_pred, svmf_pred, svm_accuracy, svmf_accuracy, svm_linear_pred, svmf_linear_pred, svm_linear_accuracy, svmf_linear_accuracy, svm_poly_pred, svmf_poly_pred, svm_poly_accuracy, svmf_poly_accuracy, svm_rbf_pred, svmf_rbf_pred, svm_rbf_accuracy, svmf_rbf_accuracy, svr_linear_mse, svr_linear_mae, svr_poly_mse, svr_poly_mae, svr_rbf_mse, svr_rbf_mae
    
