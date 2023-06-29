from functools import cache
from os.path import isfile
import random
from math import prod

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, save_npz, load_npz
from tqdm import tqdm

def get_dict(filename: str) -> dict:
    """Converts lines from the file into a dictionary of parents and indices."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for index, line in enumerate(lines):
        child, father, mother, sex = line.strip().split('\t')
        data[int(child)] = (int(father), int(mother), int(sex), index)
    return data

ADDRESS = '../data/tous_individus_pro1931-60_SAG.asc'
ABOUT = get_dict(ADDRESS)

def get_gc(proband: int) -> None:
    """Compute the genetic contributions of a given list of probands."""
    idx_map = {individual: index for index, individual in enumerate(individuals)}
    def add_gc(proband: int, ancestor: int, depth: int=0) -> None:
        if ancestor == 0:
            return
        gc_matrix[idx_map[proband], idx_map[ancestor]] += 0.5 ** depth
        male_ancestor, female_ancestor, _ = ABOUT[ancestor]
        add_gc(proband, male_ancestor, depth=depth+1)
        add_gc(proband, female_ancestor, depth=depth+1)
    add_gc(proband, proband, depth=0)

def get_gc_matrix(probands: list) -> lil_matrix:
    """Generate a matrix of genetic contributions."""
    gc_matrix = lil_matrix((len(individuals), len(individuals)), dtype=float)

    if isfile('gc_matrix.npz'):
        gc_matrix = load_npz('gc_matrix.npz')
    else:
        for proband in tqdm(probands, desc="Computing the genetic contributions"):
            get_gc(proband)
        gc_matrix = coo_matrix(gc_matrix)
        save_npz('gc_matrix.npz', gc_matrix)
    print("The matrix of genetic contributions is ready.")

    return gc_matrix

@cache
def get_kinship(individual1: int, individual2: int) -> float:
    """A recursive version of kinship coefficients from R's GENLIB library (Gauvin et al., 2015)."""
    if individual1 == individual2:
        father, mother, sex, index = ABOUT[individual1]
        if father and mother:
                value = get_kinship(father, mother)
        else:
            value = 0.
        return (1 + value) * 0.5
    
    if ABOUT[individual2][3] > ABOUT[individual1][3]:
        individual1, individual2 = individual2, individual1

    father, mother, sex, index = ABOUT[individual1]
    if not father and not mother:
        return 0.
    
    mother_value = 0.
    father_value = 0.
    if mother:
        mother_value = get_kinship(mother, individual2)
    if father:
        father_value = get_kinship(father, individual2)

    return (father_value + mother_value) / 2.

def get_probands() -> list:
    """Extracts the probands as an ordered list."""
    probands = set(ABOUT.keys()) - {parent for individual in ABOUT.keys()
                                    for parent in ABOUT[individual][:2] if parent}
    return sorted(list(probands))

def get_founders() -> list:
    """Extracts the founders as an ordered list."""
    founders = [individual for individual in ABOUT.keys()
                if not ABOUT[individual][0] and not ABOUT[individual][1]]
    founders.sort()
    return founders

def get_individuals() -> list:
    """Extracts all individuals as an ordered list."""
    individuals = list(ABOUT.keys())
    individuals.sort()
    return individuals

@cache
def get_first_spouse(individual: int) -> int:
    """Identify the first spouse of a given individual."""
    individuals = list(ABOUT.keys())
    individuals.sort()
    children = [child for child in individuals if ABOUT[child][0] == individual or ABOUT[child][1] == individual]
    first_child = children[0]
    sex = "F" if ABOUT[individual][2] == 2 else "M"
    if sex == "F":
        first_spouse = ABOUT[first_child][0]
    elif sex == "M":
        first_spouse = ABOUT[first_child][1]
    return first_spouse

@cache
def get_genotype(individual: int, carrier_founder: int) -> tuple:
    """Simulate the descent of a pathologic allele down to a population."""
    father, mother, sex, index = ABOUT[individual]
    if individual == carrier_founder:
        return (1, 0)
    if father == 0 or mother == 0:
        return (0, 0)
    fathers_genotype = get_genotype(father, carrier_founder)
    mothers_genotype = get_genotype(mother, carrier_founder)
    fathers_allele = random.choice(fathers_genotype)
    mothers_allele = random.choice(mothers_genotype)
    genotype = (fathers_allele, mothers_allele)
    return genotype

def sim_mendelian(founders: list, probands: list, individuals: list, min_homozygotes: int=10) -> tuple:
    """Simulate a Mendelian transmission of an allele from a carrier founder."""
    n_homozygotes = 0
    while n_homozygotes < min_homozygotes:
        founder = random.choice(founders)
        genotypes = [get_genotype(individual, founder) for individual in individuals]
        get_genotype.cache_clear()
        n_homozygotes = len([genotype for individual, genotype in zip(individuals, genotypes)
                             if genotype == (1, 1) and individual in probands])
    return founder, genotypes

@cache
def get_ancestors(individual: int) -> list:
    """Recursively get the ancestors of a given individual."""
    ancestors = []
    father, mother, sex, index = ABOUT[individual]
    if father != 0:
        ancestors.append(father)
        ancestors.extend(get_ancestors(father))
    if mother != 0:
        ancestors.append(mother)
        ancestors.extend(get_ancestors(mother))
    ancestors.sort()
    return ancestors

def get_common_ancestors(probands: list) -> list:
    """Get a list of all common ancestors."""
    all_ancestors = [set(get_ancestors(proband)) for proband in probands]
    common_ancestors = set.intersection(*all_ancestors)
    common_ancestors = list(common_ancestors)
    common_ancestors.sort()
    return common_ancestors

def get_homozygotes(probands: list, probands_genotypes: list) -> tuple:
    """Get a list of all homozygotes and related information."""
    hom_probands = [proband for proband, genotype in zip(probands, probands_genotypes) if genotype == (1, 1)]
    hom_genotypes = [genotype for genotype in probands_genotypes if genotype == (1, 1)]
    hom_fathers = [ABOUT[hom_proband][0] for hom_proband in hom_probands]
    hom_mothers = [ABOUT[hom_proband][1] for hom_proband in hom_probands]
    print(f"There are {len(hom_probands)} homozygotes.")
    return hom_probands, hom_genotypes, hom_fathers, hom_mothers

@cache
def get_all_paths(individual: int):
    """Get a list of all paths from a given individual to its ancestors."""
    paths = []
    father, mother, sex, index = ABOUT[individual]
    if father != 0:
        fathers_paths = get_all_paths(father)
        if len(fathers_paths) == 0:
            paths.append([father])
        for path in fathers_paths:
            new_path = path + [father]
            paths.append(new_path)
    if mother != 0:
        mothers_paths = get_all_paths(mother)
        if len(mothers_paths) == 0:
            paths.append([mother])
        for path in mothers_paths:
            new_path = path + [mother]
            paths.append(new_path)
    return paths

def get_paths(individual: int, founder: int) -> list:
    """Get a list of all paths from a proband to a given founder."""
    relevant_paths = []
    all_paths = get_all_paths(individual)
    get_all_paths.cache_clear()
    for path in all_paths:
        if founder in path:
            relevant_paths.append(path)
    return relevant_paths

def get_paths_with_parent(individual: int, parent: int, founder: int) -> list:
    """Get a list of all paths from a proband to a given founder passing by a parent."""
    relevant_paths = []
    all_paths = get_all_paths(individual)
    get_all_paths.cache_clear()
    for path in all_paths:
        if founder in path and parent in path:
            relevant_paths.append(path)
    return relevant_paths

def get_prob(ancestor: int, probands: list, probands_genotypes: list, n_samples: int=100000) -> float:
    """Calculate the probability of one ancestor to output the given genotypes."""
    prob = 0.0
    probands_paths = []
    for proband, genotype in zip(probands, probands_genotypes):
        if genotype == (0, 1) or genotype == (1, 0):
            proband_paths = get_paths(proband, ancestor)
            probands_paths.append(proband_paths)
        elif genotype == (1, 1):
            father, mother, sex, index = ABOUT[proband]
            paternal_paths = get_paths_with_parent(proband, father, ancestor)
            probands_paths.append(paternal_paths)
            maternal_paths = get_paths_with_parent(proband, mother, ancestor)
            probands_paths.append(maternal_paths)
    for _ in range(n_samples):
        chosen_paths = [random.choice(proband_paths) for proband_paths in probands_paths]
        ancestors = [set(path) for path in chosen_paths]
        unique_ancestors = set.union(*ancestors)
        prob += 0.5 ** len(unique_ancestors)
    lengths = [len(proband_paths) for proband_paths in probands_paths]
    pro = prod(lengths)
    prob = prob * pro / n_samples
    return prob

def sim_min_kinships(minor_probands: list) -> list:
    """Use a Monte-Carlo algorithm to find 10 probands with minimal kinships."""
    min_kinships = np.inf
    min_probands = []
    for _ in tqdm(range(10000), desc="Trying to find probands with low kinships"):
        possible_probands = random.choices(minor_probands, k=10)
        total_kinships = 0.0
        for pro1 in possible_probands:
            for pro2 in possible_probands:
                if pro1 <= pro2:
                    continue
                kinship = get_kinship(pro1, pro2)
                total_kinships += kinship
        if total_kinships < min_kinships:
            min_kinships = total_kinships
            min_probands = possible_probands
    min_probands.sort()

    print(f"The mean kinship is {min_kinships/10}.")

    min_genotypes = [genotype for individual, genotype in zip(individuals, genotypes) if individual in min_probands]

    return min_probands, min_genotypes

def get_probs(individuals: list, genotypes: list, n_steps: int=10) -> pd.DataFrame:
    """Calculate the probability of each ancestor to output the given genotypes."""

    probands = get_probands()
    probands_set = set(probands)
    probands_genotypes = [genotype for individual, genotype in zip(individuals, genotypes) if individual in probands_set]
    minor_probands = [proband for proband, genotype in zip(probands, probands_genotypes) if genotype != (0, 0)]
    minor_genotypes = [genotype for genotype in probands_genotypes if genotype != (0, 0)]
    print(f"There are {len(minor_probands)} probands with the minor allele.")

    min_probands, min_genotypes = sim_min_kinships(minor_probands)
    hom_probands, hom_genotypes, hom_fathers, hom_mothers = get_homozygotes(min_probands, min_genotypes)

    candidates = get_common_ancestors(min_probands + hom_fathers + hom_mothers)
    candidates_set = set(candidates)
    print(f"There are {len(candidates)} candidate ancestors.")

    candidates_genotypes = [genotype for individual, genotype in zip(individuals, genotypes)
                            if individual in candidates_set]
    candidates_spouses = [get_first_spouse(candidate) for candidate in candidates]
    spouses_genotypes = []

    idx_map = {individual: index for index, individual in enumerate(individuals)}
    for spouse in candidates_spouses:
        index = idx_map[spouse]
        genotype = genotypes[index]
        spouses_genotypes.append(genotype)

    probs = pd.DataFrame(index=candidates, columns=['Abs', 'Rel (%)', 'GT 1', 'GT 2', 'Spouse'])
    probs['Abs'] = 0.0
    probs['GT 1'] = candidates_genotypes
    probs['GT 2'] = spouses_genotypes
    probs['Spouse'] = candidates_spouses
    for _ in tqdm(range(n_steps), "Computing the probabilities"):
        for ancestor in candidates:
            prob = get_prob(ancestor, min_probands, min_genotypes)
            probs.loc[ancestor, 'Abs'] += prob / n_steps
        probs['Rel (%)'] = probs['Abs'].tolist()
        probs['Rel (%)'] /= probs['Rel (%)'].sum(axis=0)
        probs['Rel (%)'] *= 100.0
        probs.sort_values(by=['Rel (%)'], ascending=False, inplace=True)
        print(probs.head(10))
    return probs

if __name__ == '__main__':
    probands = get_probands()
    founders = get_founders()
    individuals = get_individuals()

    gc_matrix = get_gc_matrix(probands)
    matrix = csr_matrix(gc_matrix)
    means = matrix.mean(axis=0)
    selected_indices = np.argsort(means, axis=1)[0, :].tolist()[0]
    selected_indices.reverse()
    founders_set = set(founders)
    selected_founders = [individuals[index] for index in selected_indices if individuals[index] in founders_set]
    selected_founders = selected_founders[:10]

    print("Running the Mendelian simulation...")
    founder, genotypes = sim_mendelian(selected_founders, probands, individuals)
    print(f"The founder is: {founder}.")

    probs = get_probs(individuals, genotypes)
    print(probs.head(10))