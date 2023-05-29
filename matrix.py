from os import listdir

import numpy as np
from scipy.sparse import dok_matrix, save_npz
from tqdm import tqdm

def get_dict(filename: str) -> dict:
    """Converts lines from the file into a dictionary of parents and indices."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    dict = {}
    for index, line in enumerate(lines):
        child, father, mother, _ = line.strip().split('\t')
        dict[int(child)] = (int(father), int(mother), index)
    return dict

def get_probands(dict: dict) -> list:
        """Extracts the probands as an ordered list."""
        probands = set(dict.keys()) - {parent for individual in dict.keys() for parent in dict[individual][:2] if parent}
        return sorted(list(probands))

def get_unique_family_members(dict: dict) -> list:
        """Limits families to one member each."""
        visited_parents = set()
        unique_family_members = []
        probands = get_probands(dict)
        for proband in probands:
            father, mother, _ = dict[proband]
            if (father, mother) not in visited_parents:
                unique_family_members.append(proband)
                visited_parents.add((father, mother))
        return unique_family_members

dict = get_dict('../data/tous_individus_pro1931-60_SAG.asc')
unique_family_members = get_unique_family_members(dict)

matrix = dok_matrix((len(unique_family_members), len(unique_family_members)), dtype=np.float32)

for filename in tqdm(listdir('../results/kinships/'), "Compiling the DataFrame"):
    with open(f"../results/kinships/{filename}", 'r') as infile:
        for line in infile:
            proband1, proband2, kinship = line.strip().split()
            matrix[int(proband1), int(proband2)] = float(kinship)
            matrix[int(proband2), int(proband1)] = float(kinship)

matrix = matrix.tocoo()
save_npz('../results/kinships.npz', matrix)