import numpy as np
import pandas as pd

def add_noise_to_dataset(df, columns=None, noise_fraction=0.12, possible_values_dict=None):
    """
    Ajoute du bruit à plusieurs colonnes d'un DataFrame d'un seul coup.

    :param df: Le DataFrame de test original.
    :param columns: Liste des colonnes à bruiter. Si None, applique à toutes les colonnes.
    :param noise_fraction: La proportion de données à modifier par colonne (ex: 0.05).
    :param possible_values_dict: (Optionnel) Un dictionnaire {nom_colonne: [valeurs_possibles]}.
    :return: Un nouveau DataFrame avec le bruit ajouté.
    """
    df_noisy = df.copy()

    # Si aucune colonne n'est spécifiée, on prend toutes les colonnes du dataset
    if columns is None:
        columns = df_noisy.columns

    n_rows = len(df_noisy)
    n_noise = int(n_rows * noise_fraction)

    for col in columns:
        # 1. Déterminer les valeurs possibles pour cette colonne spécifique
        if possible_values_dict and col in possible_values_dict:
            possible_values = possible_values_dict[col]
        else:
            # Détection automatique : on prend les valeurs uniques existantes dans la colonne
            possible_values = df_noisy[col].dropna().unique()

        # Si la colonne est vide ou n'a qu'une seule valeur possible, on l'ignore
        if len(possible_values) <= 1:
            continue

        # 2. Sélectionner les lignes à modifier (indices au hasard)
        noise_indices = np.random.choice(df_noisy.index, size=n_noise, replace=False)

        # 3. Générer les nouvelles valeurs aléatoires parmi les choix possibles pour cette colonne
        random_new_values = np.random.choice(possible_values, size=n_noise)

        # 4. Appliquer le bruit
        df_noisy.loc[noise_indices, col] = random_new_values

    return df_noisy
