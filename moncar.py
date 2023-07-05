from functools import cache
import random
import math

from hdbscan import HDBSCAN
import pandas as pd
from tqdm import tqdm
from umap import UMAP

PEDIGREE_FILE = '../data/tous_individus_pro1931-60_SAG.asc'
ABOUT_FILE = '../data/tous_individus_dates_locations.txt'

def get_dict(filename: str) -> dict:
    """Converts lines from the file into a dictionary of parents and indices."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for index, line in enumerate(lines):
        child, father, mother, sex = line.strip().split('\t')
        data[int(child)] = (int(father), int(mother), int(sex), index)
    return data

def get_fathers(filename: str) -> dict:
    """Converts lines from the file into a dictionary of fathers."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for line in lines:
        child, father, _, _ = line.strip().split('\t')
        data[int(child)] = int(father)
    return data

def get_mothers(filename: str) -> dict:
    """Converts lines from the file into a dictionary of mothers."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for line in lines:
        child, _, mother, _ = line.strip().split('\t')
        data[int(child)] = int(mother)
    return data

def get_sexes(filename: str) -> dict:
    """Converts lines from the file into a dictionary of sexes."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for line in lines:
        child, _, _, sex = line.strip().split('\t')
        data[int(child)] = int(sex)
    return data

def get_indices(filename: str) -> dict:
    """Converts lines from the file into a dictionary of parents and indices."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for index, line in enumerate(lines):
        child, _, _, _ = line.strip().split('\t')
        data[int(child)] = index
    return data

FATHER = get_fathers(PEDIGREE_FILE)
MOTHER = get_mothers(PEDIGREE_FILE)
SEX = get_sexes(PEDIGREE_FILE)
INDEX = get_indices(PEDIGREE_FILE)

def get_city(filename: str) -> dict:
    """Converts cities from the file into a dictionary of cities and IDs."""
    data = {}
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
        for line in lines:
            line = line.strip().split('\t')
            data[int(line[0])] = int(line[4])
    return data

CITY = get_city(ABOUT_FILE)

def get_weddate(filename: str) -> dict:
    """Converts wedding years from the file into a dictionary of wedding years and IDs."""
    data = {}
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
        for line in lines:
            line = line.strip().split('\t')
            data[int(line[0])] = int(line[3])
    return data

YEAR = get_weddate(ABOUT_FILE)

"""
def get_gc(proband: int, gc_matrix: lil_matrix) -> None:
    Compute the genetic contributions of a given list of probands.
    idx_map = {individual: index for index, individual in enumerate(individuals)}
    def add_gc(ancestor: int, contribution: float) -> None:
        if not ancestor:
            return
        gc_matrix[idx_map[proband], idx_map[ancestor]] += contribution
        add_gc(FATHER[proband], contribution/2)
        add_gc(MOTHER[proband], contribution/2)
    add_gc(proband, contribution=1.0)
"""

"""
def get_gc_matrix(probands: list) -> lil_matrix:
    Generate a matrix of genetic contributions.

    ancestors = set.union(*[set(get_ancestors(proband)) for proband in probands])
    ancestors = list(ancestors)
    ancestors.sort()

    founders = [ancestor for ancestor in ancestors if not FATHER(ancestor) and not MOTHER(ancestor)]

    gc_matrix = lil_matrix((len(individuals), len(individuals)), dtype=float)
    pro_map = {proband: index for index, proband in enumerate(probands)}
    anc_map = {ancestor: index for index, ancestor in enumerate(ancestors)}

    for ancestor in tqdm(ancestors, "Filling the matrix of genetic contributions"):
        for proband in probands:
            if is_ancestor(ancestor, proband):
                gc_matrix[pro_map[proband], anc_map[ancestor]] = get_genetic_contribution(ancestor, proband)

    # if isfile('gc_matrix.npz'):
        # gc_matrix = load_npz('gc_matrix.npz')
    # else:
    for proband in tqdm(probands, desc="Computing the genetic contributions"):
        get_gc(proband, gc_matrix)
    gc_matrix = coo_matrix(gc_matrix)
    # save_npz('gc_matrix.npz', gc_matrix)
    print("The matrix of genetic contributions is ready.")

    return gc_matrix
"""

@cache
def get_genetic_contribution(fr: int, to: int) -> float:
    """Compute the genetic contribution of a single ancestor to a single descendant."""
    if fr == to:
        return 1.0
    genetic_contribution = 0.0
    father = FATHER[to]
    if father:
        if is_ancestor(fr, father):
            genetic_contribution += 0.5 * get_genetic_contribution(fr, father)
    mother = MOTHER[to]
    if mother:
        if is_ancestor(fr, mother):
            genetic_contribution += 0.5 * get_genetic_contribution(fr, mother)
    return genetic_contribution

@cache
def is_ancestor(possible_ancestor: int, individual: int) -> bool:
    """Return whether someone is the ancestor of another individual."""
    if not individual:
        return False
    if possible_ancestor == individual:
        return True
    father = FATHER[individual]
    paternal_ancestor = is_ancestor(possible_ancestor, father)
    if paternal_ancestor:
        return True
    mother = MOTHER[individual]
    maternal_ancestor = is_ancestor(possible_ancestor, mother)
    if maternal_ancestor:
        return True
    return False

@cache
def get_kinship(individual1: int, individual2: int) -> float:
    """A recursive version of kinship coefficients from R's GENLIB library (Gauvin et al., 2015)."""
    if individual1 == individual2:
        father = FATHER[individual1]
        mother = MOTHER[individual1]
        if father and mother:
                value = get_kinship(father, mother)
        else:
            value = 0.
        return (1 + value) * 0.5
    
    if INDEX[individual2] > INDEX[individual1]:
        individual1, individual2 = individual2, individual1

    father = FATHER[individual1]
    mother = MOTHER[individual1]
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
    individuals = set(INDEX.keys())
    fathers = {FATHER[individual] for individual in individuals}
    mothers = {MOTHER[individual] for individual in individuals}
    probands = individuals - fathers - mothers
    return sorted(list(probands))

def get_unique_family_members(probands: list) -> list:
        """Limits families to one member each."""
        visited_parents = set()
        unique_family_members = []
        for proband in probands:
            father = FATHER[proband]
            mother = MOTHER[proband]
            if (father, mother) not in visited_parents:
                unique_family_members.append(proband)
                visited_parents.add((father, mother))
        return unique_family_members

def get_same_origins(probands: list) -> list:
    """Limits probands to those who married at the same place as their parents."""
    same_origins = []
    for proband in probands:
        try:
            father = FATHER[proband]
            mother = MOTHER[proband]
            p_grandfather = FATHER[father]
            p_grandmother = MOTHER[father]
            m_grandfather = FATHER[mother]
            m_grandmother = FATHER[mother]
            if CITY[p_grandfather] == CITY[p_grandmother] == CITY[m_grandfather] == CITY[m_grandmother] == CITY[father] == CITY[mother] != 0:
                same_origins.append(proband)
        except KeyError:
            continue
    return same_origins

def get_founders() -> list:
    """Extracts the founders as an ordered list."""
    return sorted([individual for individual in INDEX.keys() if not FATHER[individual] and not MOTHER[individual]])

@cache
def get_first_spouse(individual: int) -> int:
    """Identify the first spouse of a given individual."""
    individuals = sorted(INDEX.keys())
    children = [child for child in individuals if FATHER[child] == individual or MOTHER[child] == individual]
    first_child = children[0]
    sex = 'F' if SEX[individual] == 2 else 'M'
    if sex == 'F':
        first_spouse = FATHER[first_child]
    elif sex == 'M':
        first_spouse = MOTHER[first_child]
    return first_spouse

@cache
def get_genotype(individual: int, carrier_founder: int) -> tuple:
    """Simulate the descent of a pathologic allele down to a population."""
    if individual == carrier_founder:
        return (1, 0)
    father = FATHER[individual]
    mother = MOTHER[individual]
    if not father or not mother:
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
    father = FATHER[individual]
    mother = MOTHER[individual]
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
    common_ancestors = sorted(common_ancestors)
    return common_ancestors

def get_homozygotes(probands: list, probands_genotypes: list) -> tuple:
    """Get a list of all homozygotes and related information."""
    hom_probands = [proband for proband, genotype in zip(probands, probands_genotypes) if genotype == (1, 1)]
    hom_genotypes = [genotype for genotype in probands_genotypes if genotype == (1, 1)]
    hom_fathers = [FATHER[hom_proband] for hom_proband in hom_probands]
    hom_mothers = [MOTHER[hom_proband] for hom_proband in hom_probands]
    print(f"There are {len(hom_probands)} homozygotes.")
    return hom_probands, hom_genotypes, hom_fathers, hom_mothers

@cache
def get_all_paths(to: int):
    """Get a list of all paths from a given individual to its ancestors."""
    paths = []
    father = FATHER[to]
    if father != 0:
        fathers_paths = get_all_paths(father)
        if len(fathers_paths) == 0:
            paths.append([father])
        for path in fathers_paths:
            new_path = path + [father]
            paths.append(new_path)
    mother = MOTHER[to]
    if mother != 0:
        mothers_paths = get_all_paths(mother)
        if len(mothers_paths) == 0:
            paths.append([mother])
        for path in mothers_paths:
            new_path = path + [mother]
            paths.append(new_path)
    return paths

"""
def select_founders(probands: list) -> list:
    Select the 100 founders with the greatest mean genetic contributions.
    gc_matrix = get_gc_matrix(probands)
    matrix = csr_matrix(gc_matrix)
    means = matrix.mean(axis=0)
    selected_indices = np.argsort(means, axis=1)[0, :].tolist()[0]
    selected_indices.reverse()
    founders = get_founders()
    selected_founders = [individuals[index] for index in selected_indices if individuals[index] in founders]
    selected_founders = selected_founders[:100]
    return selected_founders
"""

def get_paths(fr: int, to: int) -> list:
    """Get a list of all paths from a founder to a given proband."""
    relevant_paths = []
    all_paths = get_all_paths(to)
    for path in all_paths:
        if fr in path:
            relevant_paths.append(path)
    return relevant_paths

def get_paths_with_parent(fr: int, by: int, to: int) -> list:
    """Get a list of all paths from a proband to a given founder passing by a parent."""
    relevant_paths = []
    all_paths = get_all_paths(to)
    for path in all_paths:
        if fr in path and by in path:
            relevant_paths.append(path)
    return relevant_paths

def get_prob(ancestor: int, probands: list, probands_genotypes: list, n_samples: int=1000000) -> float:
    """Calculate the probability of one ancestor to output the given genotypes."""
    prob = 0.0
    probands_paths = []
    for proband, genotype in zip(probands, probands_genotypes):
        if genotype == (0, 1) or genotype == (1, 0):
            proband_paths = get_paths(proband, ancestor)
            probands_paths.append(proband_paths)
        elif genotype == (1, 1):
            father = FATHER[proband]
            paternal_paths = get_paths_with_parent(proband, father, ancestor)
            probands_paths.append(paternal_paths)
            mother = MOTHER[proband]
            maternal_paths = get_paths_with_parent(proband, mother, ancestor)
            probands_paths.append(maternal_paths)
    lengths = [len(proband_paths) for proband_paths in probands_paths]
    pro = math.prod(lengths)
    for _ in tqdm(range(n_samples)):
        chosen_paths = [random.choice(proband_paths) for proband_paths in probands_paths]
        ancestors = [set(path) for path in chosen_paths]
        unique_ancestors = set.union(*ancestors)
        prob += (0.5 ** len(unique_ancestors)) * pro
    prob = float(prob / n_samples)
    return prob

"""
def sim_min_kinships(minor_probands: list, n_probands: int=20) -> list:
    Use a Monte-Carlo algorithm to find 10 probands with minimal kinships.
    min_kinships = np.inf
    min_probands = []
    for _ in tqdm(range(10000), desc="Trying to find probands with low kinships"):
        possible_probands = random.choices(minor_probands, k=n_probands)
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
    get_kinship.cache_clear()

    print(f"The mean kinship is {min_kinships / n_probands}.")

    min_genotypes = [genotype for individual, genotype in zip(individuals, genotypes) if individual in min_probands]

    return min_probands, min_genotypes
"""

"""
def get_probs(individuals: list, genotypes: list) -> pd.DataFrame:
    Calculate the probability of each ancestor to output the given genotypes.

    probands = get_probands()
    probands_set = set(probands)
    probands_genotypes = [genotype for individual, genotype in zip(individuals, genotypes) if individual in probands_set]
    minor_probands = [proband for proband, genotype in zip(probands, probands_genotypes) if genotype != (0, 0)]
    minor_genotypes = [genotype for genotype in probands_genotypes if genotype != (0, 0)]
    print(f"There are {len(minor_probands)} probands with the minor allele.")

    min_probands, min_genotypes = sim_min_kinships(minor_probands)
    hom_probands, hom_genotypes, hom_fathers, hom_mothers = get_homozygotes(probands, probands_genotypes)

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
    for ancestor in tqdm(candidates, desc="Computing the probabilities"):
        prob = get_prob(ancestor, min_probands, min_genotypes)
        probs.loc[ancestor, 'Abs'] += prob
        probs['Rel (%)'] = probs['Abs'].tolist()
        probs['Rel (%)'] /= probs['Rel (%)'].sum(axis=0)
        probs['Rel (%)'] *= 100.0
        probs.sort_values(by=['Rel (%)'], ascending=False, inplace=True)
        print(probs.head(10))
    return probs
"""

if __name__ == '__main__':
    individuals = sorted(INDEX.keys())
    probands = get_probands()
    probands = get_unique_family_members(probands)
    probands = get_same_origins(probands)
    kinships = pd.DataFrame(index=probands, columns=probands)
    for pro1 in probands:
        for pro2 in probands:
            kinships.loc[pro1, pro2] = get_kinship(pro1, pro2)

    embedding = UMAP(n_components=20, n_neighbors=10, min_dist=0, random_state=42, verbose=True).fit_transform(kinships)
    labels = HDBSCAN(min_cluster_size=10).fit_predict(embedding)
    plot = UMAP(random_state=42, verbose=True).fit_predict(embedding)

    df = pd.DataFrame()
    df['ID'] = probands
    df['City'] = [CITY[proband] for proband in probands]
    