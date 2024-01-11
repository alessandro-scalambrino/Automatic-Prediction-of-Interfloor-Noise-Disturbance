import os
import librosa
import pandas as pd
import numpy as np
import re
from tabulate import tabulate
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

#---------------funzioni per estrarre e stampare stats---------------------

def load_excel_data(excel_path):
    return pd.read_excel(excel_path)

def extract_category(filename):
    match = re.search(r'_([^_0-9]+)', filename)
    return match.group(1) if (match and match.group(1)) else ''
    
    # Funzione per k-statics
def calculate_cohen_kappa_scores(df, pazienti):
    kappa_scores = {}
    
    for paziente1, paziente2 in combinations(pazienti, 2):
        paziente1_scores = df[paziente1]
        paziente2_scores = df[paziente2]
        kappa_score = cohen_kappa_score(paziente1_scores, paziente2_scores)
        kappa_scores[(paziente1, paziente2)] = kappa_score
    
    return kappa_scores

def calculate_patient_statistics(df, pazienti):
    media_per_paziente = df[pazienti].mean()
    deviazione_standard_per_paziente = df[pazienti].std()
    mediana_per_paziente = df[pazienti].median()
    intervallo_per_paziente = df[pazienti].max() - df[pazienti].min()
    kappa_scores = calculate_cohen_kappa_scores(df, pazienti)
    return media_per_paziente, deviazione_standard_per_paziente, mediana_per_paziente, intervallo_per_paziente, kappa_scores

def calculate_sound_statistics(df, pazienti):
    df['media_suono'] = df[pazienti].mean(axis=1)
    df['deviazione_standard_suono'] = df[pazienti].std(axis=1)
    df['mediana_suono'] = df[pazienti].median(axis=1)
    df['intervallo_suono'] = df[pazienti].max(axis=1) - df[pazienti].min(axis=1)
    return df

def calculate_category_statistics(df, pazienti):
    df['categoria'] = df['File Audio'].apply(extract_category)
    media_per_categoria = df.groupby('categoria')[pazienti].mean().reset_index()
    devstd_per_categoria = df.groupby('categoria')[pazienti].std().reset_index()
    mediana_per_categoria = df.groupby('categoria')[pazienti].median().reset_index()
    intervallo_per_categoria = (df.groupby('categoria')[pazienti].max() - df.groupby('categoria')[pazienti].min()).reset_index()

    categorie = {}
    for index, row in df.iterrows():
        categoria = row['categoria']
        voto = row['paziente1']  # Sostituisci 'paziente1' con il nome reale della colonna dei voti
        if categoria not in categorie:
            categorie[categoria] = {paziente: [] for paziente in pazienti}
        for paziente in pazienti:
            categorie[categoria][paziente].append(row[paziente])

    return media_per_categoria, devstd_per_categoria, mediana_per_categoria, intervallo_per_categoria, categorie


#stampa a schermo dei risultati


def print_patient_statistics(pazienti, media, dev_std, mediana, intervallo):
    paziente_table = pd.DataFrame({'Paziente': pazienti})
    paziente_table['Media'] = media.values
    paziente_table['Dev. Std.'] = dev_std.values
    paziente_table['Mediana'] = mediana.values
    paziente_table['Intervallo'] = intervallo.values
    print(tabulate(paziente_table, headers='keys', tablefmt='grid', showindex=False))
    print()

def print_sound_statistics(df):
    suono_table = df[['File Audio', 'media_suono', 'deviazione_standard_suono', 'mediana_suono', 'intervallo_suono']]
    concatenated_table = pd.concat([suono_table.head(5), suono_table.tail(5)])
    print(tabulate(concatenated_table, headers=suono_table.columns, tablefmt='grid', showindex=False))
    print()

def print_category_statistics(media, dev_std, mediana, intervallo, categorie, pazienti):
    print("Media di disturbo per categoria e paziente:")
    table1 = []
    for categoria, voti_per_paziente in categorie.items():
        row = [categoria]
        for paziente in pazienti:
            media_val = media.loc[media['categoria'] == categoria, paziente].values[0]
            row.append(media_val)
        table1.append(row)
    # Stampa la tabella
    headers = ['Categoria'] + pazienti
    print(tabulate(table1, headers=headers, tablefmt='grid'))
    print()

    print("Mediana di disturbo per categoria e paziente:")
    table2 = []
    for categoria, voti_per_paziente in categorie.items():
        row = [categoria]
        for paziente in pazienti:
            mediana_val = mediana.loc[mediana['categoria'] == categoria, paziente].values[0]
            row.append(mediana_val)
        table2.append(row)
    # Stampa la tabella
    print(tabulate(table2, headers=headers, tablefmt='grid'))
    print()

    print("Deviazione standard di disturbo per categoria e paziente:")
    table3 = []
    for categoria, voti_per_paziente in categorie.items():
        row = [categoria]
        for paziente in pazienti:
            dev_std_val = dev_std.loc[dev_std['categoria'] == categoria, paziente].values[0]
            row.append(dev_std_val)
        table3.append(row)
    # Stampa la tabella
    print(tabulate(table3, headers=headers, tablefmt='grid'))
    print()

    print("Intervallo di disturbo per categoria e paziente:")
    table4 = []
    for categoria, voti_per_paziente in categorie.items():
        row = [categoria]
        for paziente in pazienti:
            intervallo_val = intervallo.loc[intervallo['categoria'] == categoria, paziente].values[0]
            row.append(intervallo_val)
        table4.append(row)
    # Stampa la tabella
    print(tabulate(table4, headers=headers, tablefmt='grid'))
    print()
