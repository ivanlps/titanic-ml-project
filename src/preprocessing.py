import pandas as pd
import numpy as np

# Imputar
def impute_age(row, age_map):
    if pd.isna(row["Age"]):
        return age_map.loc[row["Title"], row["Pclass"]]
    else:
        return row["Age"]
    
def impute_deck_group(row, deck_map):
    if row["Deck"] == "U":
        key = (row["Title"], row["Pclass"])
        return deck_map.get(key, "U")
    return row["Deck"]
    
# ---------- Helpers ----------
def extract_deck(series):
    """Regresa la letra de Deck a partir de Cabin; 'U' si no hay."""
    deck = series.astype(str).str.extract(r'([A-Za-z])', expand=False)
    deck = deck.str.upper()
    return deck.fillna('U')

def most_frequent(x):
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan