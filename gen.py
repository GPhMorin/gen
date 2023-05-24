import re
from functools import cache
from os.path import exists
from itertools import combinations_with_replacement, product
from collections import deque
from time import time

import pandas as pd
from tqdm import tqdm


class Genealogy:
    """A class to handle genealogical data."""

    def __init__(self, filename: str) -> None:
        """Initializes the Gen object with a file containing genealogical data."""
        self._filename = filename
        if not exists(filename):
            raise FileNotFoundError("The pedigree file was not found.")
        with open(filename, 'r') as infile:
            lines = infile.readlines()[1:]
            self._parents = self._load_parents(lines)
            self._probands = self._extract_probands()
            self._founder = self._extract_founders()
            self._map = {individual: index for index, individual
                         in enumerate(self._parents.keys())}
            self._first_individual_paths = []
            self._second_individual_paths = []
            self._ancestors = {}

    def _load_parents(self, lines: list) -> dict:
        """Converts lines from the file into a dictionary of parents."""
        parents = {}
        for line in lines:
            # Splitting the line with multiple possible separators
            child, father, mother, _ = re.split(r'[,\t ]+', line.strip())
            father = int(father) if father != '0' else None
            mother = int(mother) if mother != '0' else None
            parents[int(child)] = (father, mother)
        return parents

    def _extract_probands(self) -> list:
        """Extracts the probands as an ordered list."""
        probands = set(self._parents.keys()) - {
            parent for ind in self._parents.keys()
            for parent in self._parents[ind]
            if parent and parent in self._parents.keys()
        }
        probands = list(probands)
        probands.sort()
        return probands

    def _extract_founders(self) -> list:
        """Extracts the founders as an ordered list."""
        founders = [
            individual for individual in self._parents.keys()
            if self.get_parents(individual) == (None, None)
        ]
        founders.sort()
        return founders

    def get_probands(self) -> list:
        """Get an ordered list of all probands."""
        return self._probands

    @cache
    def get_parents(self, child: int) -> tuple:
        """Get the parents (father, mother) of a given individual."""
        try:
            return self._parents[child]
        except KeyError:
            raise KeyError(
                f"The parents of child {child} were not found. "
                "You are not supposed to see this message."
            )

    @cache
    def get_ancestors(self, descendant: int) -> set:
        """Get all known ancestors of a given individual."""
        ancestors = set()
        stack = [descendant]
        while stack:
            current = stack.pop()
            father, mother = self.get_parents(current)
            if father and father not in ancestors:
                stack.append(father)
                ancestors.add(father)
            if mother and mother not in ancestors:
                stack.append(mother)
                ancestors.add(mother)
        ancestors.discard(None)
        return ancestors

    def get_common_ancestors(self, individuals: list) -> set:
        """Get all most-recent common ancestors (MRCAs) from a group of individuals."""
        ancestors_list = [
            self.get_ancestors(individual).union({individual})
            if individual not in self._ancestors
            else self._ancestors[individual]
            for individual in individuals
        ]
        common_ancestors = set.intersection(*ancestors_list)
        common_ancestors.discard(None)
        return common_ancestors or set()

    @cache
    def search_mrcas(self, ancestor: int, common_ancestors: list) -> set:
        """Search MRCAs in the ancestors of a given ancestor."""
        stack = [ancestor]
        mrcas = set()
        while stack:
            current = stack.pop()
            if current in common_ancestors:
                mrcas.add(current)
                continue
            father, mother = self.get_parents(current)
            if father:
                stack.append(father)
            if mother:
                stack.append(mother)
        return mrcas

    @cache
    def get_mrcas(self, individual1: int, individual2: int) -> set:
        """Get all most-recent common ancestors (MRCAs) from a group of individuals."""
        common_ancestors = self.get_common_ancestors([individual1, individual2])
        if not common_ancestors:
            return set()

        mrcas = set.union(*[
            self.search_mrcas(ancestor, common_ancestors)
            for ancestor in [individual1, individual2]
        ])

        ancestors_of_mrcas = set.union(*[
            self.get_ancestors(ancestor) for ancestor in mrcas
        ])

        return mrcas - ancestors_of_mrcas

    def shortest_path(self, df: pd.DataFrame, proband: int,
                      ancestor: int, mrca: int, depth: int) -> None:
        """Recursively find the shortest path to the MRCA."""
        if ancestor == mrca and df.at[proband, mrca] > depth:
            df.at[proband, mrca] = depth
            return

        father, mother = self.get_parents(ancestor)
        if father:
            self.shortest_path(df, proband, father, mrca, depth + 1)
        if mother:
            self.shortest_path(df, proband, mother, mrca, depth + 1)

    def get_shortest_distances(self, probands: list) -> pd.DataFrame:
        """Get the shortest distances (generations) from each proband to their most-recent common ancestors (MRCAs)."""
        all_mrcas = set()
        for individual1 in probands:
            for individual2 in probands:
                if individual1 < individual2:
                    mrca_set = self.get_mrcas(individual1, individual2)
                    if mrca_set:
                        all_mrcas.update(mrca_set)
        
        all_mrcas = sorted(list(all_mrcas))
        probands.sort()

        df = pd.DataFrame(10000, index=probands, columns=all_mrcas)

        for proband in probands:
            for mrca in all_mrcas:
                if proband == mrca:
                    df.at[proband, mrca] = 0
                    continue
                father, mother = self.get_parents(proband)
                if father:
                    self.shortest_path(df, proband, father, mrca, 1)
                if mother:
                    self.shortest_path(df, proband, mother, mrca, 1)

        df.replace(10000, None, inplace=True)
        return df
    
    def prepare_kinships(self, interests: list) -> None:
        """Prepare the files for IdCoefs (M Abney, 2009)."""
        # Load the file into a pandas DataFrame
        df = pd.read_csv(self._filename, sep='\t')
        df.set_index('ind', inplace=True)

        # Create a dictionary to store the reordered data
        reordered_data = {}

        def dfs(node):
            # If the node has already been visited, return
            if node in reordered_data:
                return
            # If the node's parents are known and haven't been visited yet, visit them first
            if df.at[node, 'father'] != 0:
                dfs(df.at[node, 'father'])
            if df.at[node, 'mother'] != 0:
                dfs(df.at[node, 'mother'])
            # Add the node to the reordered data
            reordered_data[node] = [node] + df.loc[node].tolist()

        # Start the DFS at each node
        for node in tqdm(df.index, desc="Reordering the pedigree"):
            dfs(node)

        # Convert the reordered data to a DataFrame
        reordered_df = pd.DataFrame.from_dict(reordered_data, orient='index')

        # Save the reordered DataFrame to a file
        reordered_df.to_csv('gen.pedigree', sep='\t', header=False, index=False)

        with open('gen.study', 'w') as outfile:
            combos = list(combinations_with_replacement(interests, 2))[:100]
            for individual1, individual2 in tqdm(combos, desc="Writing the individuals of interest"):
                outfile.write(f"{individual1} + {individual2}\n")

    def get_one_per_family(self) -> list:
        """Extract only one proband per family."""
        visited_parents = set()
        probands = []
        for proband in self.get_probands():
            father, mother = self.get_parents(proband)
            if (father, mother) not in visited_parents:
                probands.append(proband)
                visited_parents.add((father, mother))
        return probands
    
    def get_all_paths(self, current: int, target_ancestor: int, intermediate_ancestors: set, history: list, pathway: str) -> None:
        '''Find all possible paths from an individual to a set of common ancestors.'''
        new_history = history.copy()
        new_history.append(current)
        
        if current == target_ancestor:
            if pathway == 'individual1':
                self._first_individual_paths.append(new_history)
            elif pathway == 'individual2':
                self._second_individual_paths.append(new_history)
            return

        father, mother = self.get_parents(current)

        # Avoid cycles by not revisiting already visited nodes.
        if father and father in intermediate_ancestors and father not in history:
            self.get_all_paths(father, target_ancestor, intermediate_ancestors, new_history, pathway)
        if mother and mother in intermediate_ancestors and mother not in history:
            self.get_all_paths(mother, target_ancestor, intermediate_ancestors, new_history, pathway)
    
    @cache
    def bfs_paths(self, start: int, goal: int) -> list:
        """Find all paths from start to goal using breadth-first search."""
        queue = deque([(start, [start])])
        paths = []

        while queue:
            node, path = queue.popleft()

            if node == goal:
                paths.append(path)
            else:
                neighbors = self._parents[node]
                for neighbor in neighbors:
                    if neighbor and neighbor not in path:
                        queue.append((neighbor, path + [neighbor]))

        return paths

    @cache
    def get_inbreeding(self, individual: int) -> float:
        """Compute the coefficient of inbreeding of a given individual."""
        father, mother = self.get_parents(individual)
        if not father or not mother:
            return 0.

        common_ancestors = self.get_common_ancestors([father, mother])
        if not common_ancestors:
            return 0.
        
        coefficient = 0.
            
        for common_ancestor in list(common_ancestors):
            Fca = self.get_inbreeding(common_ancestor)
            paths1 = self.bfs_paths(father, common_ancestor)
            paths2 = self.bfs_paths(mother, common_ancestor)
            possible_loops = [path1 + path2 for path1 in paths1 for path2 in paths2]
            loops = [loop for loop in possible_loops if len(loop) - len(set(loop)) == 1]
            for loop in loops:
                coefficient += 0.5 ** len(set(loop)) * (1. + Fca)

        return coefficient
    
    def get_kinship(self, individual1: int, individual2: int) -> float:
        """Compute the coefficient of kinship between two individuals."""
        common_ancestors = self.get_common_ancestors([individual1, individual2])
        if not common_ancestors:
            return 0.
        
        if individual1 == individual2:
            Find = self.get_inbreeding(individual1)
            return 0.5 * (1 + Find)

        coefficient = 0.
        
        for common_ancestor in list(common_ancestors):
            Fca = self.get_inbreeding(common_ancestor)
            paths1 = self.bfs_paths(individual1, common_ancestor)
            paths2 = self.bfs_paths(individual2, common_ancestor)
            possible_loops = [path1 + path2 for path1 in paths1 for path2 in paths2]
            loops = [loop for loop in possible_loops if len(loop) - len(set(loop)) == 1]
            for loop in loops:
                coefficient += 0.5 ** len(set(loop)) * (1. + Fca)

        return coefficient

    def recursive_kinship(self, individual1: int, individual2: int, max1: int, max2: int) -> float:
        """A recursive version of kinship coefficients from R's GENLIB library (Gauvin et al.,2015)."""
        if individual1 == individual2:
            father, mother = self._parents[individual1]
            if father and mother:
                maximum = max(max1, max2)
                if maximum > 0:
                    value = self.recursive_kinship(father, mother, maximum-1, maximum-1)
                else:
                    value = 0.
            else:
                value = 0.

            return (1 + value) * 0.5
        
        if self._map[individual2] > self._map[individual1]:
            individual1, individual2 = individual2, individual1
            max1, max2 = max2, max1

        father, mother = self._parents[individual1]
        if not father and not mother:
            return 0.
        
        mother_value = 0.
        father_value = 0.
        if max1 > 0:
            if mother:
                mother_value = self.recursive_kinship(mother, individual2, max1-1, max2)
            if father:
                father_value = self.recursive_kinship(father, individual2, max1-1, max2)

        return (father_value + mother_value) / 2.