import pandas as pd
from functions.func_dataset_management import *
from functions.func_stat import *
from functions.func_machine_learning import *
from functions.func_plot import *




excel_path = './elenco_file_audio_backup.xlsx'
df = load_excel_data(excel_path)
audio_folder = './dataset_rand'
soggetti = ['soggetto1', 'soggetto2', 'soggetto3', 'soggetto4', 'soggetto5', 'soggetto6']

#   ANALISI STATISTICA DEI RISULTATI
# Calcola e stampa le statistiche
media_per_soggetto, deviazione_standard_per_soggetto, mediana_per_soggetto, intervallo_per_soggetto, kappa_scores = calculate_patient_statistics(df, soggetti)
df = calculate_sound_statistics(df, soggetti)
media_per_categoria, devstd_per_categoria, mediana_per_categoria, intervallo_per_categoria, categorie = calculate_category_statistics(df, soggetti)

print_patient_statistics(soggetti, media_per_soggetto, deviazione_standard_per_soggetto, mediana_per_soggetto, intervallo_per_soggetto)
print_sound_statistics(df)
print_category_statistics(media_per_categoria, devstd_per_categoria, mediana_per_categoria, intervallo_per_categoria, categorie, soggetti)

# k-statistics
for coppia, kappa_score in kappa_scores.items():
    print(f"Cohen's Kappa tra {coppia[0]} e {coppia[1]}: {kappa_score}")

df_excel = pd.read_excel(excel_path)
df = calculate_average_ratings(df_excel, excel_path, audio_folder)
# chiamata a prepare data (estrazione feature e creazione dei set di train/test)
result = prepare_data(df, audio_folder)
print("Dataframe:")
print(result)
if result is not None:
 
    X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test = result
    print("---X_train----------------")
    print(X_train)
    print("--X_test-----------------")
    print(X_test)
    print("--y_train-----------------")
    print(y_train)
    print("---y_test----------------")
    print(y_test)
    print("---F_train----------------")
    print(F_train)
    print("----F_test---------------")
    print(F_test)
    print("---z_train----------------")
    print(z_train)
    print("---z_test----------------")
    print(z_test)
    
# MACHINE LEARNING
    linear_pred, linearf_pred, mse_linear, mae_linear = linear_regression(X_train, X_test, z_train, z_test, F_train, F_test)
    knn_pred, knnf_pred, _, knn_accuracy, knnf_accuracy, mse, mae = knn_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)
    tree_pred, treef_pred, tree_accuracy, treef_accuracy, tree_reg_mse, tree_reg_mae = decision_tree_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)

    svm_pred, svmf_pred, svm_accuracy, svmf_accuracy, svm_linear_pred, svmf_linear_pred, svm_linear_accuracy, svmf_linear_accuracy, svm_poly_pred, svmf_poly_pred, svm_poly_accuracy, svmf_poly_accuracy, svm_rbf_pred, svmf_rbf_pred, svm_rbf_accuracy, svmf_rbf_accuracy, svr_linear_mse, svr_linear_mae, svr_poly_mse, svr_poly_mae, svr_rbf_mse, svr_rbf_mae = svm_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)
#plot dei risultati
    plot_results(z_test, linear_pred, linearf_pred, y_test, knn_pred, knn_accuracy, knnf_pred, knnf_accuracy,
                 tree_pred, tree_accuracy, treef_pred, treef_accuracy, svm_pred, svm_accuracy, svmf_pred,
                 svmf_accuracy, svm_linear_pred, svmf_linear_pred, svm_linear_accuracy, svmf_linear_accuracy,
                 svm_poly_pred, svmf_poly_pred, svm_poly_accuracy, svmf_poly_accuracy, svm_rbf_pred, svmf_rbf_pred,
                 svm_rbf_accuracy, svmf_rbf_accuracy, mse_linear, mae_linear, mse, mae, tree_reg_mse, tree_reg_mae, svr_linear_mse, svr_linear_mae, svr_poly_mse, svr_poly_mae, svr_rbf_mse, svr_rbf_mae)

else:
    print("Dataframe vuoto")
