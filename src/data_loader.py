from pathlib import Path
import pandas as pd

def load_and_dedup(data_dir: Path, pattern: str = '*.csv'):
    """
    Funkcja wczytuje wszystkie pliki CSV z podanego katalogu (data_dir) zgodnie z zadanym wzorcem (pattern).
    Następnie łączy je w jeden DataFrame, dodając informację o źródłowym pliku.
    Jeśli dane zawierają kolumnę 'id', usuwa duplikaty pozostawiając ostatni rekord dla każdego 'id'.
    Zwraca połączony i oczyszczony DataFrame.

    Parametry:
        data_dir (Path): Ścieżka do katalogu z plikami CSV.
        pattern (str): Wzorzec wyszukiwania plików (domyślnie '*.csv').

    Zwraca:
        pd.DataFrame: Połączony DataFrame bez duplikatów po 'id'.
    """
    # Pobierz listę plików CSV zgodnych z podanym wzorcem w katalogu data_dir
    csv_files = list(Path(data_dir).glob(pattern))
    # Jeśli nie znaleziono żadnych plików, zgłoś wyjątek
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir} (pattern={pattern})")
    df_list = []
    # Iteruj po znalezionych plikach CSV
    for file in csv_files:
        # Wczytaj plik CSV do tymczasowego DataFrame
        tmp = pd.read_csv(file)
        # Dodaj kolumnę z nazwą pliku źródłowego (bez rozszerzenia)
        tmp['source_file'] = file.stem
        # Dodaj tymczasowy DataFrame do listy
        df_list.append(tmp)
    # Połącz wszystkie DataFrame'y w jeden duży DataFrame
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(csv_files)} files. Combined shape: {df.shape}")
    # Jeśli istnieje kolumna 'id', usuń duplikaty pozostawiając ostatni rekord dla każdego 'id'
    # Sortowanie po 'id' i 'source_file' (może być nietrwałe, docelowo lepiej użyć timestamp)
    if 'id' in df.columns:
        df = df.sort_values(['id', 'source_file'])
        df = df.drop_duplicates(subset='id', keep='last').reset_index(drop=True)
        print(f"After dedup (keep last per id): {df.shape}")
    # Zwróć końcowy DataFrame
    return df