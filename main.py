import pandas as pd
from functions.func_dataset_management import *
from functions.func_stat import *
from functions.func_machine_learning import *
from functions.func_plot import *

# Load data
excel_path = './elenco_file_audio_backup.xlsx'
df = load_excel_data(excel_path)
audio_folder = './dataset_rand'
soggetti = ['soggetto1', 'soggetto2', 'soggetto3', 'soggetto4', 'soggetto5', 'soggetto6']

# Initialize a results dictionary for statistical analysis
statistics_results = {
    'patient_statistics': {},
    'sound_statistics': {},
    'category_statistics': {}
}

# Calculate patient statistics
statistics_results['patient_statistics']['media'], statistics_results['patient_statistics']['deviazione_standard'], \
statistics_results['patient_statistics']['mediana'], statistics_results['patient_statistics']['intervallo'], \
statistics_results['patient_statistics']['kappa_scores'] = calculate_patient_statistics(df, soggetti)

# Calculate sound statistics
statistics_results['sound_statistics'] = calculate_sound_statistics(df, soggetti)

# Calculate category statistics
statistics_results['category_statistics']['media'], statistics_results['category_statistics']['devstd'], \
statistics_results['category_statistics']['mediana'], statistics_results['category_statistics']['intervallo'], \
statistics_results['category_statistics']['categorie'] = calculate_category_statistics(df, soggetti)

# Centralized printing function
def print_statistics(statistics, soggetti):
    print("Patient Statistics:")
    print_patient_statistics(
        soggetti,
        statistics['patient_statistics']['media'],
        statistics['patient_statistics']['deviazione_standard'],
        statistics['patient_statistics']['mediana'],
        statistics['patient_statistics']['intervallo']
    )

    print("\nSound Statistics:")
    print_sound_statistics(statistics['sound_statistics'])

    print("\nCategory Statistics:")
    print_category_statistics(
        statistics['category_statistics']['media'],
        statistics['category_statistics']['devstd'],
        statistics['category_statistics']['mediana'],
        statistics['category_statistics']['intervallo'],
        statistics['category_statistics']['categorie'],
        soggetti
    )

# Print all statistics
print_statistics(statistics_results, soggetti)

# Kappa statistics output
for coppia, kappa_score in statistics_results['patient_statistics']['kappa_scores'].items():
    print(f"Cohen's Kappa tra {coppia[0]} e {coppia[1]}: {kappa_score}")

# Prepare data for machine learning
df_excel = pd.read_excel(excel_path)
df = calculate_average_ratings(df_excel, excel_path, audio_folder)

# Call prepare_data function to extract features and create train/test sets
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
    predictions = {}

    # Linear Regression
    predictions['linear'] = linear_regression(X_train, X_test, z_train, z_test, F_train, F_test)
    
    # KNN Classification
    predictions['knn'] = knn_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)

    # Decision Tree Classification
    predictions['tree'] = decision_tree_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)

    # SVM Classification
    predictions['svm'] = svm_classification(X_train, X_test, y_train, y_test, F_train, F_test, z_train, z_test)

    # Plot the results
    plot_results(
        z_test,
        predictions['linear'][0], predictions['linear'][1], predictions['linear'][2], predictions['linear'][3],
        predictions['knn'][0], predictions['knn'][1], predictions['knn'][2], predictions['knn'][3],
        predictions['tree'][0], predictions['tree'][1], predictions['tree'][2], predictions['tree'][3],
        predictions['svm'][0], predictions['svm'][1], predictions['svm'][2], predictions['svm'][3],
        predictions['svm'][4], predictions['svm'][5], predictions['svm'][6], predictions['svm'][7],
        predictions['svm'][8], predictions['svm'][9], predictions['svm'][10], predictions['svm'][11],
        predictions['svm'][12], predictions['svm'][13], predictions['svm'][14], predictions['svm'][15]
    )

else:
    print("Dataframe vuoto")
