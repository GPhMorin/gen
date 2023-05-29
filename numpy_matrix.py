from os import listdir

import numpy as np
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

map = {individual:index for index, individual in enumerate(unique_family_members)}

matrix = np.zeros((len(unique_family_members), len(unique_family_members)), dtype=float)

for filename in tqdm(listdir('../results/kinships/'), "Compiling the DataFrame"):
    with open(f"../results/kinships/{filename}", 'r') as infile:
        for line in infile:
            proband1, proband2, kinship = line.strip().split()
            matrix[map[int(proband1)], map[int(proband2)]] = float(kinship)
            matrix[map[int(proband2)], map[int(proband1)]] = float(kinship)

np.savez('../results/numpy_kinships.npz', matrix)