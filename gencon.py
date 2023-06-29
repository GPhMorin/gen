from functools import cache

import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix, coo_matrix, save_npz
from tqdm import tqdm

def get_dict(filename: str) -> dict:
    """Converts lines from the file into a dictionary of parents and indices."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for index, line in enumerate(lines):
        child, father, mother, _ = line.strip().split('\t')
        data[int(child)] = (int(father), int(mother), index)
    return data

dict = get_dict('../data/tous_individus_pro1931-60_SAG.asc')
ancestors = set()

@cache
def add_ancestors(individual: int) -> None:
    """Recursively add the ancestors to a set."""
    father, mother, _ = dict[individual]
    if father != 0:
        ancestors.add(father)
        add_ancestors(father)
    if mother != 0:
        ancestors.add(mother)
        add_ancestors(mother)

input_df = pd.read_csv("grands-n2d.csv")
output_df = input_df.copy()
probands = output_df['ID'].tolist()

for proband in tqdm(probands, desc="Getting a list of all ancestors"):
    add_ancestors(proband)

ancestors = list(ancestors)
ancestors.sort()

individuals = probands + ancestors
individuals.sort()

idx_map = {individual: index for index, individual in enumerate(individuals)}
gc_matrix = dok_matrix((len(individuals), len(individuals)), dtype=np.float64)

def get_gc(proband: list) -> None:
    """Compute the genetic contributions of a given list of probands."""
    def add_gc(proband: int, ancestor: int, depth: int=0) -> None:
        if ancestor == 0:
            return
        gc_matrix[idx_map[proband], idx_map[ancestor]] += 0.5 ** depth
        male_ancestor, female_ancestor, _ = dict[ancestor]
        add_gc(proband, male_ancestor, depth=depth+1)
        add_gc(proband, female_ancestor, depth=depth+1)
    add_gc(proband, proband, depth=0)

for proband in tqdm(probands, desc="Computing the genetic contributions"):
    get_gc(proband)

save_npz("gencon.npz", coo_matrix(gc_matrix))