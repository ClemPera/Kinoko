import os
import pandas as pd
from utils import add_noise_to_dataset

def get_data() -> pd.DataFrame:
    """
    Create a dataset with all tabular secondary_data, with Species name from primary_data
    Create a link with species_name to have the scientific name, so we can merge with images later on
    - load from all table datasets
    - assign species name to df_secondary
    - feature selection (facilitate user entries)
    """
    # Load data from folder
    df_primary = pd.read_csv("../data/table_dataset/primary_data.csv", sep=";")
    df_secondary = pd.read_csv("../data/table_dataset/secondary_data.csv", sep=";")
    df_mushnames = pd.read_csv("../data/table_dataset/species_names.csv", sep =";")

    # Species name from primary, add to secondary
    primary_name_df = df_primary[['family','name']]
    df_primname_rep = primary_name_df.loc[primary_name_df.index.repeat(353)].reset_index(drop=True)

    # Clean names
    data_secondary_labelled = pd.concat([df_secondary, df_primname_rep], axis=1)
    data_secondary_labelled['family'] = data_secondary_labelled['family'].str.replace(" Family", "", regex=False)
    data_secondary_labelled['Common Name'] = data_secondary_labelled["family"] + " " + data_secondary_labelled["name"]

    # Merge to scientific names
    data_merge_scname = data_secondary_labelled.merge(df_mushnames, how='left', on='Common Name')
    data_tabular_final = data_merge_scname.drop(columns=['family','name','Common Name'])
    data_tabular_final.columns = data_tabular_final.columns.str.replace('-', '_').str.replace(' ', '_').str.lower()
    data_tabular_final["class"] = (data_tabular_final["class"] == "p").astype(int)

    # Feature selection
    data_tabular_final = data_tabular_final.drop(columns=['cap_surface','ring_type',
                                                          'cap_diameter','stem_height','stem_width'])

    # Randomly ordered df
    data_tabular_rdm = data_tabular_final.sample(frac=1, random_state=3).reset_index(drop=True)

    # Apply a noise to the data : TOO perfect is not good
    data_tab_rdm_noise = add_noise_to_dataset(data_tabular_rdm, noise_fraction=0.12)

    print("✅ Tabular data loaded and cleaned, shape:", data_tabular_rdm.shape)
    return data_tab_rdm_noise

def get_data_reduced() -> pd.DataFrame:
    """
    From get_data(), keep the species that are common for tabular and images
    - load images
    - keep the species only when tabular AND image available
    """
    ## Liste champignons (via noms scientifiques) dans les images
    path_edible = "../data/image_dataset/edible"
    path_poisonous = "../data/image_dataset/poisonous"

    # Liste des sous-dossiers
    list_edible = [f for f in os.listdir(path_edible)
                     if os.path.isdir(os.path.join(path_edible, f))]
    list_poisonous = [f for f in os.listdir(path_poisonous)
                     if os.path.isdir(os.path.join(path_poisonous, f))]

    # Création du DataFrame
    df1 = pd.DataFrame(list_edible, columns=["scientific_name"])
    df1['type'] = "edible"
    df2 = pd.DataFrame(list_poisonous, columns=["scientific_name"])
    df2['type'] = "poisonous"
    df_concat = pd.concat([df1, df2], ignore_index=True)
    df_concat['scientific_name'] = df_concat['scientific_name'].str.replace("_", " ", regex=False).str.replace("-", " ", regex=False)

    # garder que le tabulaire dont l'espèce est présente dans les données d'image
    data_tabular_image = get_data().merge(df_concat, how='inner', on='scientific_name')

    return data_tabular_image
