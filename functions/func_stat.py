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
def calculate_cohen_kappa_scores(df, soggetti):
    kappa_scores = {}
    
    for soggetto1, soggetto2 in combinations(soggetti, 2):
        soggetto1_scores = df[soggetto1]
        soggetto2_scores = df[soggetto2]
        kappa_score = cohen_kappa_score(soggetto1_scores, soggetto2_scores)
        kappa_scores[(soggetto1, soggetto2)] = kappa_score
    
    return kappa_scores

def calculate_patient_statistics(df, soggetti):
    media_per_soggetto = df[soggetti].mean()
    deviazione_standard_per_soggetto = df[soggetti].std()
    mediana_per_soggetto = df[soggetti].median()
    intervallo_per_soggetto = df[soggetti].max() - df[soggetti].min()
    kappa_scores = calculate_cohen_kappa_scores(df, soggetti)
    return media_per_soggetto, deviazione_standard_per_soggetto, mediana_per_soggetto, intervallo_per_soggetto, kappa_scores

def calculate_sound_statistics(df, soggetti):
    df['media_suono'] = df[soggetti].mean(axis=1)
    df['deviazione_standard_suono'] = df[soggetti].std(axis=1)
    df['mediana_suono'] = df[soggetti].median(axis=1)
    df['intervallo_suono'] = df[soggetti].max(axis=1) - df[soggetti].min(axis=1)
    return df

def calculate_category_statistics(df, soggetti):
    df['categoria'] = df['File Audio'].apply(extract_category)
    media_per_categoria = df.groupby('categoria')[soggetti].mean().reset_index()
    devstd_per_categoria = df.groupby('categoria')[soggetti].std().reset_index()
    mediana_per_categoria = df.groupby('categoria')[soggetti].median().reset_index()
    intervallo_per_categoria = (df.groupby('categoria')[soggetti].max() - df.groupby('categoria')[soggetti].min()).reset_index()

    categorie = {}
    for index, row in df.iterrows():
        categoria = row['categoria']
        voto = row['soggetto1']  
        if categoria not in categorie:
            categorie[categoria] = {soggetto: [] for soggetto in soggetti}
        for soggetto in soggetti:
            categorie[categoria][soggetto].append(row[soggetto])

    return media_per_categoria, devstd_per_categoria, mediana_per_categoria, intervallo_per_categoria, categorie



def print_patient_statistics(soggetti, media, dev_std, mediana, intervallo):
    soggetto_table = pd.DataFrame({'Paziente': soggetti})
    soggetto_table['Media'] = media.values
    soggetto_table['Dev. Std.'] = dev_std.values
    soggetto_table['Mediana'] = mediana.values
    soggetto_table['Intervallo'] = intervallo.values
    print(tabulate(soggetto_table, headers='keys', tablefmt='grid', showindex=False))
    print()

def print_sound_statistics(df):
    suono_table = df[['File Audio', 'media_suono', 'deviazione_standard_suono', 'mediana_suono', 'intervallo_suono']]
    concatenated_table = pd.concat([suono_table.head(5), suono_table.tail(5)])
    print(tabulate(concatenated_table, headers=suono_table.columns, tablefmt='grid', showindex=False))
    print()

def print_category_statistics(media, dev_std, mediana, intervallo, categorie, soggetti):
    print("Media di disturbo per categoria e paziente:")
    table1 = []
    for categoria, voti_per_soggetto in categorie.items():
        row = [categoria]
        for soggetto in soggetti:
            media_val = media.loc[media['categoria'] == categoria, soggetto].values[0]
            row.append(media_val)
        table1.append(row)
    headers = ['Categoria'] + soggetti
    print(tabulate(table1, headers=headers, tablefmt='grid'))
    print()

    print("Mediana di disturbo per categoria e paziente:")
    table2 = []
    for categoria, voti_per_soggetto in categorie.items():
        row = [categoria]
        for soggetto in soggetti:
            mediana_val = mediana.loc[mediana['categoria'] == categoria, soggetto].values[0]
            row.append(mediana_val)
        table2.append(row)
    print(tabulate(table2, headers=headers, tablefmt='grid'))
    print()

    print("Deviazione standard di disturbo per categoria e paziente:")
    table3 = []
    for categoria, voti_per_soggetto in categorie.items():
        row = [categoria]
        for soggetto in soggetti:
            dev_std_val = dev_std.loc[dev_std['categoria'] == categoria, soggetto].values[0]
            row.append(dev_std_val)
        table3.append(row)
    print(tabulate(table3, headers=headers, tablefmt='grid'))
    print()

    print("Intervallo di disturbo per categoria e paziente:")
    table4 = []
    for categoria, voti_per_soggetto in categorie.items():
        row = [categoria]
        for soggetto in soggetti:
            intervallo_val = intervallo.loc[intervallo['categoria'] == categoria, soggetto].values[0]
            row.append(intervallo_val)
        table4.append(row)
    print(tabulate(table4, headers=headers, tablefmt='grid'))
    print()
