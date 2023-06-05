from geo import MRC, mrc_dict

import numpy as np
from umap import UMAP
import plotly.express as px

def get_dict(filename: str) -> dict:
    """Converts lines from the file into a dictionary of parents and indices."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for index, line in enumerate(lines):
        child, father, mother, _ = line.strip().split('\t')
        data[int(child)] = (int(father), int(mother), index)
    return data

def get_probands(data: dict) -> list:
        """Extracts the probands as an ordered list."""
        probands = set(data.keys()) - {parent for individual in data.keys() for parent in data[individual][:2] if parent}
        return sorted(list(probands))

def get_unique_family_members(data: dict, probands: list) -> list:
        """Limits families to one member each."""
        visited_parents = set()
        unique_family_members = []
        for proband in probands:
            father, mother, _ = data[proband]
            if (father, mother) not in visited_parents:
                unique_family_members.append(proband)
                visited_parents.add((father, mother))
        return unique_family_members

def get_city(filename: str) -> dict:
    """Converts cities from the file into a dictionary of cities and IDs."""
    data = {}
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
        for line in lines:
            line = line.strip().split('\t')
            data[int(line[0])] = int(line[4])
    return data

def get_same_origins(data: dict, probands: list) -> list:
    "Limits probands to those who married at the same place as their parents."
    same_origins = []
    city = get_city('../data/tous_individus_dates_locations.txt')
    for proband in probands:
        try:
            father, mother, _ = data[proband]
            p_grandfather, p_grandmother, _ = data[father]
            m_grandfather, m_grandmother, _ = data[mother]
            if city[p_grandfather] == city[p_grandmother] and city[m_grandfather] == city[m_grandmother]:
                if mrc_dict[city[p_grandfather]] == mrc_dict[city[m_grandmother]]:
                    same_origins.append(proband)
        except KeyError:
            continue
    return same_origins

data = get_dict('../data/tous_individus_pro1931-60_SAG.asc')
probands = get_probands(data)
unique_family_members = get_unique_family_members(data, probands)
same_origins = get_same_origins(data, unique_family_members)
city = get_city('../data/tous_individus_dates_locations.txt')
mrcs = [mrc_dict[city[data[data[individual][0]][0]]] for individual in same_origins]

print("Preparing the labels")
y = np.array(mrcs)
y_names = {
    MRC.HORS_MRC: "Mashteuiatsh",
    MRC.LAC_SAINT_JEAN_EST: "Lac-Saint-Jean-Est",
    MRC.LE_DOMAINE_DU_ROY: "Le Domaine-du-Roy",
    MRC.LE_FJORD_DU_SAGUENAY: "Le Fjord-du-Saguenay",
    MRC.MARIA_CHAPDELAINE: "Maria-Chapdelaine",
    MRC.SAGUENAY: "Saguenay"
}

X = np.load('anchorgae-emb.npy')

print("Reducing the dimensionality...")
emb = UMAP(verbose=True, random_state=42).fit_transform(X)

print("Generating the scatter plot...")
fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=[y_names[y_value] for y_value in y], hover_name=same_origins)
fig.write_html('plotly-gae.html')