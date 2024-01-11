import pandas as pd
from functions.func_dataset_management import *
from functions.func_stat import *
from functions.func_machine_learning import *
from functions.func_plot import *




excel_path = '../elenco_file_audio_backup.xlsx'
df = load_excel_data(excel_path)
audio_folder = './dataset_rand'
pazienti = ['paziente1', 'paziente2', 'paziente3', 'paziente4', 'paziente5', 'paziente6']

# Calcola e stampa le statistiche
media_per_paziente, deviazione_standard_per_paziente, mediana_per_paziente, intervallo_per_paziente, kappa_scores = calculate_patient_statistics(df, pazienti)
df = calculate_sound_statistics(df, pazienti)
media_per_categoria, devstd_per_categoria, mediana_per_categoria, intervallo_per_categoria, categorie = calculate_category_statistics(df, pazienti)

print_patient_statistics(pazienti, media_per_paziente, deviazione_standard_per_paziente, mediana_per_paziente, intervallo_per_paziente)
print_sound_statistics(df)
print_category_statistics(media_per_categoria, devstd_per_categoria, mediana_per_categoria, intervallo_per_categoria, categorie, pazienti)

# k-statistics
for coppia, kappa_score in kappa_scores.items():
    print(f"Cohen's Kappa tra {coppia[0]} e {coppia[1]}: {kappa_score}")

df_excel = pd.read_excel(excel_path)
df = calculate_average_ratings(df_excel, excel_path, audio_folder)

result = prepare_data(df, audio_folder)
if result is not None:
    X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test = result

    linear_pred, linearf_pred = linear_regression(X_train, X_test, z_train, z_test, F_train, F_test)
    knn_pred, knnf_pred, _, knn_accuracy, knnf_accuracy = knn_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)
    tree_pred, treef_pred, tree_accuracy, treef_accuracy = decision_tree_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)

    svm_pred, svmf_pred, svm_accuracy, svmf_accuracy, svm_linear_pred, svmf_linear_pred, svm_linear_accuracy, svmf_linear_accuracy, svm_poly_pred, svmf_poly_pred, svm_poly_accuracy, svmf_poly_accuracy, svm_rbf_pred, svmf_rbf_pred, svm_rbf_accuracy, svmf_rbf_accuracy = svm_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)

    plot_results(z_test, linear_pred, linearf_pred, y_test, knn_pred, knn_accuracy, knnf_pred, knnf_accuracy,
                 tree_pred, tree_accuracy, treef_pred, treef_accuracy, svm_pred, svm_accuracy, svmf_pred,
                 svmf_accuracy, svm_linear_pred, svmf_linear_pred, svm_linear_accuracy, svmf_linear_accuracy,
                 svm_poly_pred, svmf_poly_pred, svm_poly_accuracy, svmf_poly_accuracy, svm_rbf_pred, svmf_rbf_pred,
                 svm_rbf_accuracy, svmf_rbf_accuracy)

else:
    print("Dataframe vuoto")
