import os
import librosa
import librosa.display
import pandas as pd



def create_dataframe(excel_path, audio_folder, df_media_voti):
    # Leggi il file Excel
    df_excel = pd.read_excel(excel_path)

    # Creazione di un DataFrame per i dati includendo solo i file WAV
    data = {'file': [], 'label': []}
    for file in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, file)
        if file.lower().endswith('.wav') and os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            data['file'].append(file_name)
            # Estrai la media dei voti corrispondente al file dalla tabella df_media_voti
            file_label = df_media_voti[df_media_voti['File Audio'] == file]['Media_Voti'].values
            
            # Assicurati che il file abbia una corrispondenza nella tabella df_media_voti
            if len(file_label) > 0:
                data['label'].append(file_label[0])
            else:
                # Puoi gestire il caso in cui non ci sia una corrispondenza, ad esempio assegnando un valore di default
                data['label'].append('valore_di_default')

    # Crea e restituisci il DataFrame
    return pd.DataFrame(data)


