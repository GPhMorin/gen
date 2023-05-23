import pandas as pd
from tqdm import tqdm
import re
from functools import cache
from os.path import exists

class Gen:
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
            self._map = {individual:index for index, individual
                        in enumerate(self._parents.keys())}
            self._ind1_paths = []
            self._ind2_paths = []
            self._ancestors = {}
            
    def _load_parents(self, lines: list) -> dict:
        """Converts lines from the file into a dictionary of parents."""
        parents = {}
        for line in lines:
            # Splitting the line with multiple possible separators
            ind, father, mother, _ = re.split(r'[,\t ]+', line.strip())
            father = int(father) if father != '0' else None
            mother = int(mother) if mother != '0' else None
            parents[int(ind)] = (father, mother)
        return parents

    def _extract_probands(self) -> list:
        """Extracts the probands as an ordered list."""
        probands = set(self._parents.keys()) - {parent for ind in self._parents.keys()
                                           for parent in self._parents[ind]
                                           if parent and parent in self._parents.keys()}
        probands = list(probands)
        probands.sort()
        return probands
    
    def _extract_founders(self) -> list:
        """Extracts the founders as an ordered list."""
        founders = [individual for individual in self._parents.keys()
                    if self.get_parents(individual) == (None, None)]
        founders.sort()
        return founders
    
    def get_probands(self) -> list:
        """Get an ordered list of all probands."""
        return self._probands

    @cache
    def get_parents(self, ind: int) -> tuple:
        """Get the parents (father, mother) of a given individual."""
        try:
            return self._parents[ind]
        except KeyError:
            return (None, None)

    @cache
    def get_ancestors(self, ind: int) -> set:
        """Recursively get all known ancestors of a given individual."""
        if not ind:
            return set()

        father, mother = self.get_parents(ind)
        ancestors = {father, mother}

        if father:
            ancestors |= self.get_ancestors(father)
        if mother:
            ancestors |= self.get_ancestors(mother)

        return ancestors - {None}

    def get_common_ancestors(self, individuals: list) -> set:
        """Get all most-recent common ancestors (MRCAs) from a group of individuals."""
        ancestors_list = [self.get_ancestors(individual).union({individual})
                          if individual not in self._ancestors 
                          else self._ancestors[individual] 
                          for individual in individuals]
        common_ancestors = set.intersection(*ancestors_list)
        common_ancestors.discard(None)
        return common_ancestors or None
    
    def search_mrcas(self, ancestor: int, common_ancestors: list) -> set:
        """Search MRCAs in the ancestors of a given ancestor."""
        if ancestor in common_ancestors:
            return {ancestor}
        
        father, mother = self.get_parents(ancestor)
        return (self.search_mrcas(father, common_ancestors) if father else set()) | \
               (self.search_mrcas(mother, common_ancestors) if mother else set())

    def get_mrcas(self, ind1: int, ind2: int) -> set:
        """Recursively get all most-recent common ancestors (MRCAs) from a group of individuals."""
        common_ancestors = self.get_common_ancestors([ind1, ind2])
        if not common_ancestors:
            return set()

        mrca_set = set.union(*[self.search_mrcas(ancestor, common_ancestors)
                               for ancestor in [ind1, ind2]])
        
        ancestors_of_mrcas = set.union(*[self.get_ancestors(ancestor) for ancestor in list(mrca_set)])

        return mrca_set - ancestors_of_mrcas

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
        for ind1 in probands:
            for ind2 in probands:
                if ind1 < ind2:
                    mrca_set = self.get_mrcas(ind1, ind2)
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
        df = pd.read_csv(self._filename, sep=' ')
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
            reordered_data[node] = [node] + df.at[node].tolist()

        # Start the DFS at each node
        for node in tqdm(df.index, desc="Reordering the pedigree"):
            dfs(node)

        # Convert the reordered data to a DataFrame
        reordered_df = pd.DataFrame.from_dict(reordered_data, orient='index')

        # Save the reordered DataFrame to a file
        reordered_df.to_csv('gen.pedigree', sep='\t', header=False, index=False)

        with open('gen.study', 'w') as outfile:
            for interest in tqdm(interests, desc="Writing the individuals of interest"):
                outfile.write(f"{interest}\n")

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
            if pathway == 'ind1':
                self._ind1_paths.append(new_history)
            elif pathway == 'ind2':
                self._ind2_paths.append(new_history)
            return

        father, mother = self.get_parents(current)

        # Avoid cycles by not revisiting already visited nodes.
        if father and father in intermediate_ancestors and father not in history:
            self.get_all_paths(father, target_ancestor, intermediate_ancestors, new_history, pathway)
        if mother and mother in intermediate_ancestors and mother not in history:
            self.get_all_paths(mother, target_ancestor, intermediate_ancestors, new_history, pathway)
    
    @cache
    def get_inbreeding(self, individual: int) -> float:
        """Compute the coefficient of inbreeding of a given individual."""
        father, mother = self.get_parents(individual)
        if not father or not mother:
            return 0.

        common_ancestors = self.get_common_ancestors([father, mother])
        history = [individual]

        if not common_ancestors:
            return 0.
        
        coeff = 0.
        
        for common_ancestor in list(common_ancestors):
            fathers_ancestors = self.get_ancestors(father)
            mothers_ancestors = self.get_ancestors(mother)
            ancestors_ancestors = self.get_ancestors(common_ancestor)

            self.get_all_paths(father, common_ancestor, fathers_ancestors-ancestors_ancestors, history, 'ind1')
            self.get_all_paths(mother, common_ancestor, mothers_ancestors-ancestors_ancestors, history, 'ind2')

            ind1_paths = self._ind1_paths.copy()
            self._ind1_paths.clear()
            ind2_paths = self._ind2_paths.copy()
            self._ind2_paths.clear()

            for ind1_path in ind1_paths:
                for ind2_path in ind2_paths:
                    loop = ind1_path + ind2_path[::-1][1:]
                    if len(loop) - len(set(loop)) != 1:
                        continue
                    Fca = self.get_inbreeding(common_ancestor)
                    coeff += 0.5 ** (len(loop) - 2.) * (1. + Fca)

        return coeff
    
    def get_kinship(self, ind1: int, ind2: int) -> float:
        """Compute the coefficient of kinship between two individuals."""
        common_ancestors = self.get_common_ancestors([ind1, ind2])
        if not common_ancestors:
            return 0.
        
        if ind1 == ind2:
            Find = self.get_inbreeding(ind1)
            return 0.5 * (1 + Find)

        coeff = 0.
        history = []
        
        for common_ancestor in list(common_ancestors):
            ind1_ancestors = self.get_ancestors(ind1)
            ind2_ancestors = self.get_ancestors(ind2)
            ancestors_ancestors = self.get_ancestors(common_ancestor)

            self.get_all_paths(ind1, common_ancestor, ind1_ancestors-ancestors_ancestors, history, 'ind1')
            self.get_all_paths(ind2, common_ancestor, ind2_ancestors-ancestors_ancestors, history, 'ind2')

            ind1_paths = self._ind1_paths.copy()
            self._ind1_paths.clear()
            ind2_paths = self._ind2_paths.copy()
            self._ind2_paths.clear()

            for ind1_path in ind1_paths:
                for ind2_path in ind2_paths:
                    loop = ind1_path + ind2_path[::-1][1:]
                    if len(loop) - len(set(loop)) != 0:
                        continue
                    Fca = self.get_inbreeding(common_ancestor)
                    coeff += 0.5 ** len(loop) * (1. + Fca)

        return coeff