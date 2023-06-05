from enum import Enum

from scipy.sparse import load_npz
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

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

def get_sedentary_families(data: dict, probands: list) -> list:
    "Limits probands to those who married at the same place as their parents."
    sedentary_probands = []
    city = get_city('../data/tous_individus_dates_locations.txt')
    for proband in probands:
         father, mother, _ = data[proband]
         try:
              probands_city = city[proband]
              fathers_city = city[father]
              mothers_city = city[mother]
              p_grandfather, p_grandmother, _ = data[father]
              m_grandfather, m_grandmother, _ = data[mother]
              p_grandfathers_city = city[p_grandfather]
              p_grandmothers_city = city[p_grandmother]
              m_grandfathers_city = city[m_grandfather]
              m_grandmothers_city = city[m_grandmother]
         except KeyError:
              continue
         if probands_city == fathers_city == mothers_city == p_grandfathers_city == p_grandmothers_city == m_grandfathers_city == m_grandmothers_city:
              sedentary_probands.append(proband)
    return sedentary_probands

def convert_indices(data: dict) -> dict:
     """Return the index of a sibling if that sibling is the unique family member."""
     idx_map = {}
     probands = get_probands(data)
     unique_family_members = get_unique_family_members(data, probands)
     visited_parents = {}
     for proband in probands:
          father, mother, _ = data[proband]
          if (father, mother) not in visited_parents:
               index = unique_family_members.index(proband)
               idx_map[proband] = index
               visited_parents[(father, mother)] = index
          else:
               index = visited_parents[(father, mother)]
               idx_map[proband] = index
     return idx_map

data = get_dict('../data/tous_individus_pro1931-60_SAG.asc')
probands = get_probands(data)
sedentary_probands = get_sedentary_families(data, probands)
unique_family_members = get_unique_family_members(data, sedentary_probands)
idx_map = convert_indices(data)
indices = [idx_map[proband] for proband in unique_family_members]
city = get_city('../data/tous_individus_dates_locations.txt')
cities = [city[individual] for individual in unique_family_members]

class MRC(Enum):
    LE_DOMAINE_DU_ROY = 910
    MARIA_CHAPDELAINE = 920
    LAC_SAINT_JEAN_EST = 930
    SAGUENAY = 94068
    LE_FJORD_DU_SAGUENAY = 942
    HORS_MRC = 1

class City(Enum):
    ALBANEL = 2758
    ALMA = 2759
    L_ANSE_SAINT_JEAN = 2761
    ARVIDA = 2762
    L_ASCENSION_DE_NOTRE_SEIGNEUR = 2763
    LA_BAIE = 2765
    BEGIN = 2771
    CHAMBORD = 2778
    CHICOUTIMI = 2779
    CHICOUTIMI_NORD = 2780
    CHUTE_DES_PASSES = 2782
    DESBIENS = 2786
    DOLBEAU = 2787
    FERLAND_ET_BOILEAU = 2790
    GIRARDVILLE = 2794
    HEBERTVILLE = 2799
    HEBERTVILLE_STATION = 2800
    JONQUIERE = 2804
    LAC_BOUCHETTE = 2806
    LA_DORE = 2808
    LAROUCHE = 2811
    LATERRIERE = 2812
    MISTASSINI = 2822
    NORMANDIN = 2826
    NOTRE_DAME_DE_LORETTE = 2827
    NOTRE_DAME_DU_ROSAIRE = 2832
    PERIBONKA = 2834
    PETIT_SAGUENAY = 2835
    MASHTEUIATSH = 2839
    RIVIERE_ETERNITE = 2848
    ROBERVAL = 2853
    SAINT_AMBROISE = 2860
    SAINT_ANDRE_DU_LAC_SAINT_JEAN = 2863
    SAINT_AUGUSTIN = 2869
    SAINT_BRUNO = 2871
    SAINT_CHARLES_DE_BOURGET = 2872
    DELISLE = 2876
    SAINT_DAVID_DE_FALARDEAU = 2881
    SAINT_EDMOND = 2885
    SAINT_EUGENE_D_ARGENTENAY = 2890
    SAINT_FELICIEN = 2895
    SAINT_FELIX_D_OTIS = 2896
    SAINT_FRANCOIS_DE_SALES = 2898
    SAINT_FULGENCE = 2902
    SAINT_GEDEON = 2905
    SAINT_HENRI_DE_TAILLON = 2910
    SAINT_HONORE = 2913
    METABETCHOUAN = 2921
    LABRECQUE = 2927
    SAINT_LUDGER_DE_MILOT = 2931
    SAINT_NAZAIRE = 2940
    SAINT_PRIME = 2951
    SAINT_STANISLAS = 2956
    SAINT_THOMAS_DIDYME = 2958
    LAC_A_LA_CROIX = 2969
    SAINTE_ELISABETH_DE_PROULX = 2970
    SAINTE_HEDWIDGE = 2977
    SAINTE_JEANNE_D_ARC = 2981
    SAINT_METHODE = 2984
    SAINTE_MONIQUE = 2985
    SAINTE_ROSE_DU_NORD = 2989
    SHIPSHAW = 2992
    MONT_APICA = 3005

mrc = {
     City.ALBANEL.value: MRC.MARIA_CHAPDELAINE.value,
     City.ALMA.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.L_ANSE_SAINT_JEAN.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.ARVIDA.value: MRC.SAGUENAY.value,
     City.L_ASCENSION_DE_NOTRE_SEIGNEUR.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.LA_BAIE.value: MRC.SAGUENAY.value,
     City.BEGIN.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.CHAMBORD.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.CHICOUTIMI.value: MRC.SAGUENAY.value,
     City.CHICOUTIMI_NORD.value: MRC.SAGUENAY.value,
     City.CHUTE_DES_PASSES.value: MRC.MARIA_CHAPDELAINE.value,
     City.DESBIENS.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.DOLBEAU.value: MRC.MARIA_CHAPDELAINE.value,
     City.FERLAND_ET_BOILEAU.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.GIRARDVILLE.value: MRC.MARIA_CHAPDELAINE.value,
     City.HEBERTVILLE.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.HEBERTVILLE_STATION.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.JONQUIERE.value: MRC.SAGUENAY.value,
     City.LAC_BOUCHETTE.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.LA_DORE.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.LAROUCHE.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.LATERRIERE.value: MRC.SAGUENAY.value,
     City.MISTASSINI.value: MRC.MARIA_CHAPDELAINE.value,
     City.NORMANDIN.value: MRC.MARIA_CHAPDELAINE.value,
     City.NOTRE_DAME_DE_LORETTE.value: MRC.MARIA_CHAPDELAINE.value,
     City.NOTRE_DAME_DU_ROSAIRE.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.PERIBONKA.value: MRC.MARIA_CHAPDELAINE.value,
     City.PETIT_SAGUENAY.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.MASHTEUIATSH.value: MRC.HORS_MRC.value,
     City.RIVIERE_ETERNITE.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.ROBERVAL.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.SAINT_AMBROISE.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.SAINT_ANDRE_DU_LAC_SAINT_JEAN.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.SAINT_AUGUSTIN.value: MRC.MARIA_CHAPDELAINE.value,
     City.SAINT_BRUNO.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINT_CHARLES_DE_BOURGET.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.DELISLE.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINT_DAVID_DE_FALARDEAU.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.SAINT_EDMOND.value: MRC.MARIA_CHAPDELAINE.value,
     City.SAINT_EUGENE_D_ARGENTENAY.value: MRC.MARIA_CHAPDELAINE.value,
     City.SAINT_FELICIEN.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.SAINT_FELIX_D_OTIS.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.SAINT_FRANCOIS_DE_SALES.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.SAINT_FULGENCE.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.SAINT_GEDEON.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINT_HENRI_DE_TAILLON.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINT_HONORE.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.METABETCHOUAN.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.LABRECQUE.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINT_LUDGER_DE_MILOT.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINT_NAZAIRE.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINT_PRIME.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.SAINT_STANISLAS.value: MRC.MARIA_CHAPDELAINE.value,
     City.SAINT_THOMAS_DIDYME.value: MRC.MARIA_CHAPDELAINE.value,
     City.LAC_A_LA_CROIX.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINTE_ELISABETH_DE_PROULX.value: MRC.MARIA_CHAPDELAINE.value,
     City.SAINTE_HEDWIDGE.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.SAINTE_JEANNE_D_ARC.value: MRC.MARIA_CHAPDELAINE.value,
     City.SAINT_METHODE.value: MRC.LE_DOMAINE_DU_ROY.value,
     City.SAINTE_MONIQUE.value: MRC.LAC_SAINT_JEAN_EST.value,
     City.SAINTE_ROSE_DU_NORD.value: MRC.LE_FJORD_DU_SAGUENAY.value,
     City.SHIPSHAW.value: MRC.SAGUENAY.value,
     City.MONT_APICA.value: MRC.LAC_SAINT_JEAN_EST.value
}

print("Loading the matrix...")
matrix = load_npz('../results/kinships.npz')

print("Transforming the sparse matrix to an array...")
X = matrix.toarray()

print("Preparing the labels")
y = np.array([mrc[city] for city in cities])
y_names = {
     MRC.HORS_MRC.value: "Mashteuiatsh",
     MRC.LAC_SAINT_JEAN_EST.value: "Lac-Saint-Jean-Est",
     MRC.LE_DOMAINE_DU_ROY.value: "Le Domaine-du-Roy",
     MRC.LE_FJORD_DU_SAGUENAY.value: "Le Fjord-du-Saguenay",
     MRC.MARIA_CHAPDELAINE.value: "Maria-Chapdelaine",
     MRC.SAGUENAY.value: "Saguenay"
}

''' Useful when not selecting probands
print("Removing empty data...")
indices = np.where(np.sum(matrix, axis=1) == 0.5)[0]
X = np.delete(X, indices, axis=0)
X = np.delete(X, indices, axis=1)
y = np.delete(y, indices, axis=0)
'''

'''
probands = get_probands(dict)
original_probands = get_unique_family_members(dict, probands)
indices = [index for index, original_proband in enumerate(original_probands) if original_proband not in sedentary_probands]
X = np.delete(X, indices, axis=0)
X = np.delete(X, indices, axis=1)
'''

'''
indices_saguenay = [index for index, mrc in enumerate(y) if mrc == MRC.SAGUENAY.value]
X = np.delete(X, indices_saguenay, axis=0)
X = np.delete(X, indices_saguenay, axis=1)
y = np.delete(y, indices_saguenay, axis=0)
'''

X = X[indices, :]
X = X[:, indices]

print(X.shape)
print(len(y))

print("Reducing the dimensionality...")
pca = PCA(n_components=0.95, random_state=42).fit_transform(X)
print(pca.shape)
emb = UMAP(verbose=True, random_state=42).fit_transform(pca)

print("Generating the scatter plot...")
sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=[y_names[y_value] for y_value in y])
plt.savefig("TSNE.png")