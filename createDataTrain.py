import yfinance as yf

def scarica_dati_e_salva_csv(ticker, inizio, fine, intervallo, nome_file_csv):
    """
    Scarica i dati storici per un dato titolo (ticker) da Yahoo Finance e li salva in un file CSV.

    :param ticker: Il simbolo del titolo da scaricare (es. 'AAPL' per Apple).
    :param inizio: Data di inizio nel formato 'YYYY-MM-DD'.
    :param fine: Data di fine nel formato 'YYYY-MM-DD'.
    :param intervallo: L'intervallo dei dati ('1d', '1wk', '1mo', etc.).
    :param nome_file_csv: Il nome del file CSV in cui salvare i dati.
    :return: None
    """
    # Scarica i dati
    dati = yf.download(ticker, start=inizio, end=fine, interval=intervallo)

    # Salva i dati in un file CSV
    dati.to_csv(nome_file_csv)

# Esempio di utilizzo della funzione
scarica_dati_e_salva_csv('AAPL', '2023-11-21', '2023-12-21', '1d', 'datasets/dataTrain.csv')