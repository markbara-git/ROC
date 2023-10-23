import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve, auc
from scipy import interp
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import interp
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score




# Wczytanie pliku CSV
df = pd.read_csv('input-we.csv')
#df = pd.read_csv('input-0.csv')

# Wypełnienie pustych miejsc zerami
df = df.fillna(0)

# Z pliku csv wybieramy 5 kolumn
df = df[['geopixel','name','cve_css_total_score', 'device_total_security_score',\
         'osi_model_layer1_security_posture_assessment']]

# Posortuj dane według kolumny 'geopixel'
df['geopixel'] = df['geopixel'].astype(str)
df_sorted = df.sort_values(by='geopixel')

# Nadpisz oryginalny DataFrame
df = df.sort_values(by='geopixel')

# Zapisz zmieniony DataFrame do pliku CSV
df.to_csv('input-sort.csv', index=False)

# Wyświetl początkowe wiersze posortowanego DataFrame, aby sprawdzić wynik
#print(df.head())

# Utworzenie nowej kolumny "WYNIK" i wypełnienie jej zerami
df['WYNIK'] = 0

# Zapisanie zmodyfikowanego pliku CSV z zerami
df.to_csv('input-we02.csv', index=False)

# Konwersja kolumny "score" na typ float
df['cve_css_total_score'] = pd.to_numeric(df['cve_css_total_score'], errors='coerce')

# Przeskalowanie kolumny "cve_css_total_score"
df['cve_css_total_score'] = df['cve_css_total_score']* 0.1

# Zapisanie zmienionego DataFrame do pliku CSV
df.to_csv('input-we03.csv', index=False)


# Wczytanie danych z pliku CSV
df = pd.read_csv('input-we03.csv')
print(df.head())

# Definiujemy zmienną wybrane_kolumny, które chcemy uwzględnić w warunku
wybrane_kolumny = ["cve_css_total_score"]
                   
# Wybieramy wiersze, w których wartości w wybranych kolumnach są większe od 0.6
warunek = (df["device_total_security_score"].astype(float) > 0.6) 

# Przypisanie wartości 1 do kolumny "WYNIK" dla wierszy spełniających warunek
df.loc[warunek, 'WYNIK'] = 1

# Zapisanie zmienionego DataFrame do pliku CSV
df.to_csv("input-we04.csv", index=False)
print(df.head())



# Podzial pliku CSV (input-we04.csv) na czesci po 100 rekordow/ lub inny podzial
chunk_size = 100
for i, chunk in enumerate(pd.read_csv('input-we06.csv', chunksize=chunk_size)):
    chunk.to_csv(f'czesc_{i}.csv', index=False)
    chunk.dropna(inplace=True)

# Petla przez wszystkie czesci zbioru
for i in range(100):
    # Wczytanie czesci danych
    chunk = pd.read_csv(f"czesc_{i}.csv")
    chunk = chunk.fillna(0)

# sciezka do katalogu, w ktorym znajduja sie pliki czesciowe
folder_path = "C:/Users/jtancula/Modyly/"

# Inicjalizacja pustych list do przechowywania wynikow
geopixel_list = []
max_auc_list = []
macierz_pomylek = []

# Główna pętla  (będzie działać, dopóki są nowe pliki częściowe w katalogu)
while True:
    file_list = [f for f in os.listdir(folder_path) if f.startswith('czesc_') and f.endswith('.csv')]
    
    # Jeśli nie ma żadnych plików, zatrzymaj pętlę
    if not file_list:
        break
    
    # Iteracja po plikach częściowych
    for nazwa_pliku in file_list:
        df_czesc = pd.read_csv(os.path.join(folder_path, nazwa_pliku))
        
        # X - kolumny z cechami 
        X = df_czesc[['geopixel','cve_css_total_score', 'device_total_security_score', \
                      'osi_model_layer1_security_posture_assessment']]  
        
        # y- kolumna z etykietami
        y = df_czesc['WYNIK']  
        
        # Inicjalizacja i uczenie modelu logistycznego
        model = LogisticRegression()
        model.fit(X, y)
        
        # Predykcja etykiet na danych treningowych
        y_pred = model.predict(X)
        
        # Obliczenie macierzy pomyłek
        macierz = confusion_matrix(y, y_pred)
        
        # Obliczenie pola AUC
        auc = roc_auc_score(y, y_pred)

        # Pobranie unikalnych wartości 'geopixel' dla danej części
        unique_geopixels_in_part = df_czesc['geopixel'].unique()

        for geopixel_value in unique_geopixels_in_part:
            # Dodaj wynik tylko, jeśli geopixel nie istnieje jeszcze w liście
            if geopixel_value not in geopixel_list:
                # Dodanie wyników do list
                geopixel_list.append(geopixel_value)
                max_auc_list.append(auc)
                macierz_pomylek.append(macierz)
            else:
                # Jeśli geopixel już istnieje, zaktualizuj wartość AUC tylko jeśli jest większa
                index = geopixel_list.index(geopixel_value)
                if auc > max_auc_list[index]:
                    max_auc_list[index] = auc
        
        # Wyświetlenie wyników
        print("Macierz pomyłek:" ,macierz)
        print(f"Geopixel: {geopixel_list[-1]}, AUC: {auc}")

    # Po przetworzeniu plików usuwamy je z katalogu, aby uniknąć ich ponownego przetwarzania
    for nazwa_pliku in file_list:
        sciezka_pliku = os.path.join(folder_path, nazwa_pliku)
        os.remove(sciezka_pliku)
    
    # Należy zaczekać i sprawdzić, czy pojawiły się nowe pliki
    time.sleep(3)  # Dostosowanie czasu oczekiwania w sekundach

# Utworzenie listy wyników z formatem ('Geopixel', 'Najwyższe Pole AUC')
wyniki = list(zip(geopixel_list, max_auc_list))

# Przygotowanie ramki danych z unikalnymi geopixelami i polami AUC czyli 'geopixel','security level'
df = pd.DataFrame(wyniki, columns=['geopixel', 'security_level'])
df.to_csv('output.csv', index=False, quoting=csv.QUOTE_NONE)