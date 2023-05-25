from functools import cache
from sys import argv

def get_dict(filename: str) -> dict:
    """Converts lines from the file into a dictionary of parents and indices."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    dict = {}
    for index, line in enumerate(lines):
        # Splitting the line with multiple possible separators
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

dict = get_dict('../data/tous_individus_pro1931-60_SAG.asc')
unique_family_members = get_unique_family_members(dict)

pro1 = unique_family_members[int(argv[1])]
with open(f'../results/kinships/kinships_{pro1}', 'w') as outfile:
    for pro2 in unique_family_members:
        if pro1 <= pro2:
            kinship = get_kinship(pro1, pro2)
            outfile.write(f'{pro1} {pro2} {kinship}\n')