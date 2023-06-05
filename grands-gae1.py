from geo import mrc_dict

from scipy.sparse import load_npz
from scipy.io import savemat
import numpy as np

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

def get_weddate(filename: str) -> dict:
    """Converts wedding years from the file into a dictionary of wedding years and IDs."""
    data = {}
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
        for line in lines:
            line = line.strip().split('\t')
            data[int(line[0])] = int(line[3])
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
indices = [index for index, proband in enumerate(unique_family_members) if proband in same_origins]
city = get_city('../data/tous_individus_dates_locations.txt')
mrcs = [mrc_dict[city[individual]] for individual in same_origins]
weddate = get_weddate('../data/tous_individus_dates_locations.txt')

print("WEDDING YEARS")
print(f"Probands: {np.array([weddate[proband] for proband in same_origins]).mean()}")
parents = []
grandparents = []
for proband in same_origins:
    father, mother, _ = data[proband]
    parents.extend([father, mother])
    if father in data:
        father_grandparents = data[father][:2]
        grandparents.extend([grandparent for grandparent in father_grandparents if grandparent])
    if mother in data:
        mother_grandparents = data[mother][:2]
        grandparents.extend([grandparent for grandparent in mother_grandparents if grandparent])
print(f"Parents: {np.array([weddate[parent] for parent in parents]).mean()}")
print(f"Grandparents: {np.array([weddate[grandparent] for grandparent in grandparents]).mean()}")

print("Loading the matrix...")
matrix = load_npz('../results/kinships.npz')

print("Transforming the sparse matrix to an array...")
X = matrix.toarray()

print("Preparing the labels")
y = np.array(mrcs).reshape(-1, 1)

X = X[indices, :]
X = X[:, indices]

print(X.shape)
print(len(y))

print("Exporting the data...")
mdic = {'X': X, 'Y': y}
savemat('balsac.mat', mdic)