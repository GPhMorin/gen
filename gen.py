import pandas as pd
from tqdm import tqdm
import re
from functools import cache

class Gen:
    """A class to handle genealogical data."""

    def __init__(self, filename: str):
        """Initializes the Gen object with a file containing genealogical data."""
        self.filename = filename
        with open(filename, 'r') as infile:
            lines = infile.readlines()[1:]
            self.parents = self._load_parents(lines)
            self.pro = self._extract_probands()
            self.founders = self._extract_founders()
            self.map = {individual:index for index, individual
                        in enumerate(self.parents.keys())}
            self.paths = []
            self.loops = []
            
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
        probands = set(self.parents.keys()) - {parent for ind in self.parents.keys()
                                               for parent in self.parents[ind]
                                               if parent and parent in self.parents.keys()}
        probands = list(probands)
        probands.sort()
        return probands
    
    def _extract_founders(self) -> list:
        """Extracts the founders as an ordered list."""
        founders = [individual for individual in self.parents.keys()
                    if self.get_parents(individual) == (None, None)]
        founders.sort()
        return founders
    
    def get_probands(self) -> list:
        """Get an ordered list of all probands."""
        return self.pro

    def get_parents(self, ind: int) -> tuple:
        """Get the parents (father, mother) of a given individual."""
        try:
            return self.parents[ind]
        except KeyError:
            return (None, None)

    @cache
    def get_ancestors(self, ind: int) -> set:
        """Recursively get all known ancestors of a given individual."""
        if not ind:
            return set()

        father, mother = self.get_parents(ind)
        ancestors = {father, mother}
        fathers_ancestry = self.get_ancestors(father)
        if fathers_ancestry:
            ancestors |= fathers_ancestry
        mothers_ancestry = self.get_ancestors(mother)
        if mothers_ancestry:
            ancestors |= mothers_ancestry

        return ancestors

    def get_common_ancestors(self, individuals: list) -> set:
        """Get all most-recent common ancestors (MRCAs) from a group of individuals."""
        ancestors_list = [self.get_ancestors(individual).union({individual})
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
        if ancestor == mrca and df.loc[proband, mrca] > depth:
            df.loc[proband, mrca] = depth
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
                    df.loc[proband, mrca] = 0
                    continue
                father, mother = self.get_parents(proband)
                if father:
                    self.shortest_path(df, proband, father, mrca, 1)
                if mother:
                    self.shortest_path(df, proband, mother, mrca, 1)

        df.replace(10000, None, inplace=True)
        return df

    def all_paths(self, df: pd.DataFrame, proband: int, ancestor: int, target_ancestor: int, depth: int) -> None:
        """Recursively find all paths from a proband to its most-recent common ancestor (MRCA)."""
        if ancestor in self.visited:
            return
        else:
            self.visited.add(ancestor)

        if ancestor == target_ancestor:
            if isinstance(df.loc[proband, target_ancestor], list):
                df.loc[proband, target_ancestor].append(depth)
            else:
                df.loc[proband, target_ancestor] = [depth]

        father, mother = self.get_parents(ancestor)
        if father:
            self.all_paths(df, proband, father, target_ancestor, depth + 1)
        if mother:
            self.all_paths(df, proband, mother, target_ancestor, depth + 1)

    def prepare_kinships(self, interests: list) -> None:
        """Prepare the files for IdCoefs (M Abney, 2009)."""
        # Load the file into a pandas DataFrame
        df = pd.read_csv(self.filename, sep=' ')
        df.set_index('ind', inplace=True)

        # Create a dictionary to store the reordered data
        reordered_data = {}

        def dfs(node):
            # If the node has already been visited, return
            if node in reordered_data:
                return
            # If the node's parents are known and haven't been visited yet, visit them first
            if df.loc[node, 'father'] != 0:
                dfs(df.loc[node, 'father'])
            if df.loc[node, 'mother'] != 0:
                dfs(df.loc[node, 'mother'])
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
            for interest in tqdm(interests, desc="Writing the individuals of interest"):
                outfile.write(f"{interest}\n")

    def get_one_per_family(self) -> list:
        """Extract only one proband per family."""
        visited_parents = set()
        probands = []
        for proband in self.pro:
            father, mother = self.get_parents(proband)
            if (father, mother) not in visited_parents:
                probands.append(proband)
                visited_parents.add((father, mother))
        return probands
    
    def get_all_paths(self, current: int, target: int, history: list) -> None:
        '''Find all possible paths from an individual to a common ancestor.'''
        new_history = history.copy()
        new_history.append(current)
        
        if current == target:
            self.paths.append(new_history)
            return

        father, mother = self.get_parents(current)

        # Avoid cycles by not revisiting already visited nodes.
        if father and father not in history:
            self.get_all_paths(father, target, new_history)
        if mother and mother not in history:
            self.get_all_paths(mother, target, new_history)
    
    def get_inbreeding(self, individual: int, in_process: set = None) -> float:
        """Compute the coefficient of inbreeding of a given individual."""
        if in_process is None:
            in_process = set()

        if individual in in_process:
            return 0.

        in_process.add(individual)

        father, mother = self.get_parents(individual)
        if not father or not mother:
            return 0.

        common_ancestors = self.get_common_ancestors([father, mother])
        history = [individual]

        if not common_ancestors:
            return 0.

        for common_ancestor in common_ancestors:
            if common_ancestor == individual:
                continue

            self.get_all_paths(father, common_ancestor, history)
            self.get_all_paths(mother, common_ancestor, history)

        paths = self.paths.copy()
        for path1 in paths:
            for path2 in paths:
                if path1[1] == father and path2[1] == mother and path2[-1] == path1[-1]:
                    loop = path1.copy()
                    path2 = path2.copy()
                    path2.reverse()
                    common_ancestor = path2[0]
                    path2 = path2[1:]
                    loop.extend(path2)
                    if len(loop[1:-1]) != len(set(loop[1:-1])):
                        continue
                    else:
                        self.loops.append((common_ancestor, loop))

        coeff = 0.
        loops = self.loops.copy()
        for loop in loops:
            common_ancestor, sequence = loop
            Fca = self.get_inbreeding(common_ancestor, in_process=in_process)
            coeff += 0.5 ** (len(sequence) - 2.) * (1. + Fca)

        self.paths.clear()
        self.loops.clear()

        return coeff