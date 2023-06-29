from functools import cache
import random
import math

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.sparse import load_npz, csr_matrix
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

@cache
def get_ancestors(individual: int) -> list:
    "Recursively get the ancestors of a given individual."
    ancestors = []
    father, mother, _ = dict[individual]
    if father != 0:
        ancestors.append(father)
        ancestors.extend(get_ancestors(father))
    if mother != 0:
        ancestors.append(mother)
        ancestors.extend(get_ancestors(mother))
    return ancestors

@cache
def get_kinship(individual1: int, individual2: int) -> float:
    """A recursive version of kinship coefficients from R's GENLIB library (Gauvin et al., 2015)."""
    if individual1 == individual2:
        father, mother, _ = dict[individual1]
        if father and mother:
                value = get_kinship(father, mother)
        else:
            value = 0.
        return (1 + value) * 0.5
    
    if dict[individual2][2] > dict[individual1][2]:
        individual1, individual2 = individual2, individual1

    father, mother, _ = dict[individual1]
    if not father and not mother:
        return 0.
    
    mother_value = 0.
    father_value = 0.
    if mother:
        mother_value = get_kinship(mother, individual2)
    if father:
        father_value = get_kinship(father, individual2)

    return (father_value + mother_value) / 2.

input_df = pd.read_csv("grands-n2d.csv")
output_df = input_df.copy()
labels = ['N/A' if math.isnan(label) else '  ' + str(int(label)) if label < 10 else ' ' + str(int(label)) if label < 100 else str(int(label)) for label in output_df['Label'].tolist()]
output_df['Label'] = labels
probands = output_df['ID'].tolist()

for proband in tqdm(probands, desc="Getting a list of all ancestors"):
    add_ancestors(proband)

ancestors = list(ancestors)
ancestors.sort()

individuals = probands + ancestors
individuals.sort()

gc_matrix = load_npz("gencon.npz")
gc_matrix = csr_matrix(gc_matrix)
idx_map = {individual: index for index, individual in enumerate(individuals)}

@cache
def get_genotype(individual: int, carrier_founder: int) -> tuple:
    """Simulate the descent of a pathologic allele down to a population."""
    father, mother, _ = dict[individual]
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

sorted_indices = np.argsort(gc_matrix.mean(axis=0))
index = sorted_indices[0, -4]
carrier_ancestor = individuals[index]
carrier_founder = carrier_ancestor
father, mother, _ = dict[carrier_founder]
while father != 0 and mother != 0:
    carrier_founder = random.choice((father, mother))
    father, mother, _ = dict[carrier_founder]
mean_gc = gc_matrix.mean(axis=0)[0, index]
print(f"Carrier founder: {carrier_founder}.")
print(f"Mean genetic contribution of greatest contributor: {mean_gc}.")

group = output_df.groupby('Label')
unique_genotypes = ['(1, 1)', '(1, 0)', '(0, 1)', '(0, 0)']
index = pd.MultiIndex.from_product([group.groups.keys(), unique_genotypes], names=['Label', 'Genotype'])
total_counts = pd.Series(0, index=index, dtype=int, name='Count')
max_homozygotes = 1
max_genotypes = []

genotypes = []
greater_or_equal = False
patience = 1000
while patience:
    genotypes = [get_genotype(proband, carrier_founder) for proband in probands]
    get_genotype.cache_clear()
    homozygotes = len([genotype for genotype in genotypes if genotype == (1, 1)])
    if homozygotes >= max_homozygotes:
        max_homozygotes = homozygotes
        max_genotypes = genotypes
        patience = 100
    else:
        patience -= 1

print(f"The maximum number of homozygotes is: {max_homozygotes}")
output_df['Genotype'] = [str(genotype) for genotype in max_genotypes]
counts = group['Genotype'].value_counts()
total_counts = total_counts.add(counts, fill_value=0)

total_counts = total_counts.reset_index()
total_counts = pd.DataFrame(total_counts)
total_counts.columns = ['Label', 'Genotype', 'Count']

sums = total_counts.groupby('Label')['Count'].transform('sum')
total_counts['Ratio'] = total_counts['Count'] / sums

labels = [f"{community} (n={int(n)})" for community, n in zip(total_counts['Label'].tolist(), sums)]
total_counts['Community'] = labels

relevant_communities = [row['Label'] for _, row in total_counts.iterrows()
                        if 0 < row.loc['Ratio'] < 1 and row['Label'] != 'N/A']
relevant_individuals = [row['ID'] for _, row in output_df.iterrows()
                        if row['Label'] in relevant_communities]
minor_individuals = [row['ID'] for _, row in output_df.iterrows()
                     if row['Genotype'] != '(0, 0)']
major_individuals = [individual for individual in relevant_individuals
                     if individual not in minor_individuals]
all_ancestors = [set(get_ancestors(individual))
                 for individual in minor_individuals]
common_ancestors = set.intersection(*all_ancestors)
possible_carriers = list(common_ancestors)
possible_carriers = [possible_carrier for possible_carrier in possible_carriers
                     if dict[possible_carrier][0] == 0 and dict[possible_carrier][1] == 0]

kinships = pd.DataFrame(index=minor_individuals, columns=possible_carriers)
for minor_individual in tqdm(minor_individuals, desc="Computing the kinships"):
    for possible_carrier in possible_carriers:
        kinships.loc[minor_individual, possible_carrier] = get_kinship(minor_individual, possible_carrier)

column_means = kinships.mean()
highest_mean = column_means.max()
candidates = column_means[column_means == highest_mean].index.tolist()
print("Candidates:")
print(candidates)


"""
print("Generating the histogram...")
fig = px.histogram(
    total_counts,
    x='Community',
    color='Genotype',
    y='Ratio',
    title=f"Carrier founder: {carrier_founder}"
)

fig.write_html(f"mensim-{carrier_founder}.html")
"""

"""
relevant_communities = [row['Label'] for _, row in total_counts.iterrows()
                        if 0 < row.loc['Ratio'] < 1 and row['Label'] != 'N/A']
relevant_individuals = [row['ID'] for _, row in output_df.iterrows()
                        if row['Label'] in relevant_communities]
minor_individuals = [row['ID'] for _, row in output_df.iterrows()
                     if row['Genotype'] != '(0, 0)']
major_individuals = [individual for individual in relevant_individuals
                     if individual not in minor_individuals]

all_ancestors = [set(get_ancestors(individual))
                 for individual in minor_individuals]
common_ancestors = set.intersection(*all_ancestors)

possible_carriers = list(common_ancestors)
possible_carriers = [possible_carrier for possible_carrier in possible_carriers
                     if dict[possible_carrier][0] == 0 and dict[possible_carrier][1] == 0]

print(f"There are {len(possible_carriers)} possible carriers in the minor group.")
answer = "Yes" if carrier_founder in possible_carriers else "No"
print(f"Is the carrier founder there? {answer}.")
"""

"""
hamming = {ancestor: np.inf for ancestor in possible_carriers}
for possible_carrier in tqdm(possible_carriers, desc="Common ancestors"):
    print(f"Possible founder: {possible_carrier}")
    for i in tqdm(range(10), desc="Simulations"):
        sim_genotypes = []
        homozygotes = 0
        lesser_or_equal = False
        patience = 10
        while patience:
            sim_genotypes = [get_genotype(proband, possible_carrier) for proband in probands]
            get_genotype.cache_clear()
            homozygotes = len([genotype for genotype in genotypes if genotype == (1, 1)])
            sim_genotypes_bool = [True if genotype != (0, 0) else False for genotype in sim_genotypes]
            max_genotypes_bool = [True if genotype != (0, 0) else False for genotype in max_genotypes]
            difference = [sim_genotype for sim_genotype, max_genotype in zip(sim_genotypes_bool, max_genotypes_bool) if sim_genotype != max_genotype]
            distance = len(difference)
            if distance < hamming[possible_carrier]:
                hamming[possible_carrier] = distance
                patience = 10
            else:
                patience -= 1
    print(hamming[possible_carrier])

series = pd.Series(hamming)
series.sort_values(inplace=True)
print(series.to_dict())
"""
"""
relevant_communities = set(relevant_communities)
relevant_communities = list(relevant_communities)
relevant_communities.sort()
communities_ancestors = []
"""

"""
for community in relevant_communities:
    all_ancestors = [set(get_ancestors(row['ID']))
                     for _, row in output_df.iterrows()
                     if row['Label'] == community]
    common_ancestors = set.intersection(*all_ancestors)
    community_size = len(output_df[output_df['Label'] == community])
    communities_ancestors.append((common_ancestors, community_size))
"""
# candidates = {candidate: 1 for candidate in possible_carriers}

"""
for candidate in candidates.keys():
    for community_ancestors, community_size in communities_ancestors:
        if candidate in community_ancestors:
            candidates[candidate] += 1.0 / community_size
"""

"""
for candidate in candidates.keys():
    for index, row in output_df.iterrows():
        if candidate in get_ancestors(row['ID']) and row['Label'] in relevant_communities and row['Genotype'] != '(0, 0)':
            community_size = len(output_df[output_df['Label'] == row['Label']])
            contribution = gc_matrix[idx_map[row['ID']], idx_map[candidate]]
            candidates[candidate] += contribution / math.log(community_size)
"""

"""
for individual in minor_individuals:
    ancestors = [ancestor for ancestor in get_ancestors(individual) if ancestor in candidates.keys()]
    for ancestor in ancestors:
        contribution = gc_matrix[idx_map[individual], idx_map[ancestor]]
        candidates[ancestor].append(contribution)
    
for candidate in candidates.keys():
    candidates[candidate] = statistics.variance(candidates[candidate])
"""

"""
series = pd.Series(candidates)
series.sort_values(inplace=True)
maximum = series.max()
candidates = [candidate for candidate in candidates.keys() if candidates[candidate] == maximum]

print("Candidates:")
print(candidates)

answer = "Yes" if carrier_founder in candidates else "No"
print(f"Is the carrier founder there? {answer}.")
"""

"""
while len(major_individuals) > 0:
    print(len(major_individuals))
    all_ancestors = [set(get_ancestors(individual))
                     for individual in minor_individuals + major_individuals]
    common_ancestors = set.intersection(*all_ancestors)
    common_ancestors = list(common_ancestors)
    common_ancestors.sort()

    if len(common_ancestors) > 0:
        break

    removed_individual = random.choice(major_individuals)
    major_individuals = set(major_individuals) - {removed_individual}
    major_individuals = list(major_individuals)
    major_individuals.sort()

ancestors_ancestors = [set(get_ancestors(ancestor)) for ancestor in common_ancestors]
ancestors_ancestors = set.union(*ancestors_ancestors)
common_ancestors = set(common_ancestors)
most_recent = common_ancestors - ancestors_ancestors
most_recent = list(most_recent)
most_recent.sort()

print("Common ancestors:")
print(most_recent)
print(f"n = {len(minor_individuals + major_individuals)}")
"""