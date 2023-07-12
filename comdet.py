from functools import cache
import itertools
import statistics

import colorcet as cc
from hdbscan import HDBSCAN
import pandas as pd
import plotly.express as px
from scipy.sparse import load_npz
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from umap import UMAP

PEDIGREE_FILE = '../data/tous_individus_pro1931-60_SAG.asc'
ABOUT_FILE = '../data/tous_individus_dates_locations.txt'
LOCATION_FILE = '../data/lieux_mariage_definition.csv'

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
    """Converts lines from the file into a dictionary of indices."""
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

def get_city_codes(filename: str) -> dict:
    """Converts city codes from the file into a dictionary of city codes."""
    data = {}
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
        for line in lines:
            line = line.strip().split('\t')
            data[int(line[0])] = int(line[4])
    return data

CITYCODE = get_city_codes(ABOUT_FILE)

def get_cities(filename: str) -> dict:
    """Converts cities from the file into a dictionary of cities."""
    data = {}
    df = pd.read_csv(filename, usecols=[0, 1, 2], encoding='cp1252')
    df.columns = ['name', 'code', 'region']
    for index, row in df.iterrows():
        if row.loc['code'] == 'UrbIdMariage':
            continue
        data[int(row.loc['code'])] = row.loc['name']
    return data

CITY = get_cities(LOCATION_FILE)

def get_regions(filename: str) -> dict:
    """Converts regions from the file into a dictionary of regions."""
    data = {}
    df = pd.read_csv(filename, usecols=[0, 1, 2], encoding='cp1252')
    df.columns = ['name', 'code', 'region']
    for _, row in df.iterrows():
        if row.loc['code'] == 'UrbIdMariage':
            continue
        data[int(row.loc['code'])] = row.loc['region']
    data[16674] = 'Nouveau-Brunswick'
    data[20228] = 'Nouveau-Brunswick'
    data[16915] = 'Ontario'
    return data

REGION = get_regions(LOCATION_FILE)

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

def get_first_sibling(proband: int) -> int:
    """Returns the oldest sibling of an individual."""
    father = FATHER[proband]
    mother = MOTHER[proband]
    individuals = sorted(INDEX.keys())
    for individual in individuals:
        if FATHER[individual] == father and MOTHER[individual] == mother:
            return individual

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
            if CITYCODE[p_grandfather] == CITYCODE[p_grandmother] == CITYCODE[m_grandfather] == CITYCODE[m_grandmother] == CITYCODE[father] == CITYCODE[mother] != 0:
                same_origins.append(proband)
        except KeyError:
            continue
    return same_origins

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

def test_significance(df: pd.DataFrame) -> pd.DataFrame:
    clusters = list(df['Label'].unique())
    number_before = len(clusters)

    print("Calculating if the clusters are statistically significant...")
    if 'N/A' in clusters:
        clusters.remove('N/A')
    tests = []
    for cluster1 in tqdm(clusters, desc="Grouping the cluster combinations"):
        for cluster2 in clusters:
            if cluster1 >= cluster2:
                continue
            individuals1 = df[df['Label'] == cluster1]['ID'].tolist()
            individuals2 = df[df['Label'] == cluster2]['ID'].tolist()
            intracluster_combinationsA = itertools.combinations(individuals1, 2)
            intracluster_combinationsB = itertools.combinations(individuals2, 2)
            intercluster_combinations = itertools.product(individuals1, individuals2)
            intracluster_kinshipsA = [kinships.loc[individual1, individual2]
                                    for individual1, individual2 in intracluster_combinationsA]
            intracluster_kinshipsB = [kinships.loc[individual1, individual2]
                                    for individual1, individual2 in intracluster_combinationsB]
            intercluster_kinships = [kinships.loc[individual1, individual2]
                                    for individual1, individual2 in intercluster_combinations]
            _, p_valueA = mannwhitneyu(intracluster_kinshipsA, intercluster_kinships, alternative='greater')
            _, p_valueB = mannwhitneyu(intracluster_kinshipsB, intercluster_kinships, alternative='greater')
            tests.append((cluster1, cluster2, p_valueA))
            tests.append((cluster2, cluster1, p_valueB))
    rejects, _, _, _ = multipletests([value[2] for value in tests], alpha=0.05, method='holm')

    clusters_map = {}
    rejectA = None
    for rejectB, (cluster1, cluster2, p_value) in zip(rejects, tests):
        if rejectA is None:
            rejectA = rejectB
        else:
            if not rejectA and not rejectB:
                common_value = clusters_map.get(cluster1, cluster1)
                clusters_map[cluster1] = common_value
                clusters_map[cluster2] = common_value
            rejectA = None       

    for label in df['Label'].unique():
        if label not in clusters_map:
            clusters_map[label] = label
            
    df['Label'] = df['Label'].map(clusters_map)

    number_after = len(df['Label'].unique())

    if number_after == number_before:
        print("All clusters are statistically significant!")
    else:
        print(f"{number_after} clusters are statistically significant.")

    return df

def plot_scatter_clusters(df: pd.DataFrame) -> None:
    """Export the clusters' scatter plot."""
    print("Generating the clusters' scatter plot...")
    fig = px.scatter(
        df.sort_values(by='Label'),
        x='UMAP 1',
        y='UMAP 2',
        color='Label',
        hover_name='ID',
        hover_data=['City', 'Region'],
        template='simple_white',
        color_discrete_sequence=cc.glasbey_dark
    )
    fig.write_html('Scatter_Clusters.html')

def plot_cities_sunburst(df: pd.DataFrame) -> None:
    """Export the cities' sunburst diagram."""
    print("Generating the cities' sunburst diagram...")
    ids = []
    names = []
    parents = []
    for _, row in df.iterrows():
        proband = row['Label/ID']
        name = row['ID']
        label = row['City/Label']
        ids.append(proband)
        names.append(name)
        parents.append(label)
    for label in sorted(df['City/Label'].unique()):
        name = label.split('/')[1]
        city = label.split('/')[0]
        ids.append(label)
        names.append(name)
        parents.append(city)
    for city in sorted(df['City'].unique()):
        region = df.loc[df['City'] == city, 'Region'].iloc[0]
        ids.append(city)
        names.append(city)
        parents.append(region)
    for region in sorted(df['Region'].unique()):
        ids.append(region)
        names.append(region)
        if region == "Saguenay-Lac-St-Jean":
            parents.append("")
        else:
            parents.append("Other Regions")
    ids.append("Other Regions")
    names.append("Other Regions")
    parents.append("")

    labels = []
    previous_city = None
    previous_label = None
    for _, row in df.sort_values(by=['City', 'Label']).iterrows():
        city = row['City']
        label = row['Label']
        if row['City'] != previous_city:
            labels.append(label)
        elif label != previous_label:
            labels.append(label)
        previous_city = city
        previous_label = label

    print(labels)
        
    label_counts = df.sort_values(by='City').groupby('Label')['Value'].sum().reset_index(name='Sum')

    city_max_label = df.groupby(['City', 'Label'])['Value'].sum().reset_index()
    city_max_label = city_max_label.loc[city_max_label.groupby('City')['Value'].idxmax()][['City', 'Label']]
    city_counts = df.sort_values(by='Label').groupby('City')['Value'].sum().reset_index(name='Sum')

    region_max_label = df.groupby(['Region', 'Label'])['Value'].sum().reset_index()
    region_max_label = region_max_label.loc[region_max_label.groupby('Region')['Value'].idxmax()][['Region', 'Label']]
    region_counts = df.sort_values(by='City').groupby('Region')['Value'].sum().reset_index(name='Sum')

    notsag_df = df[df['Region'] != "Saguenay-Lac-St-Jean"].groupby(['Region', 'Label'])['Value'].sum().reset_index()
    notsag_max_label = notsag_df.loc[notsag_df.groupby('Region')['Value'].idxmax()]['Label'].iloc[0]
    notsag_count = df[df['Region'] != "Saguenay-Lac-St-Jean"]['Value'].sum()

    colors = df['Label'].tolist() + labels + city_max_label['Label'].tolist() + region_max_label['Label'].tolist() + [notsag_max_label]
    values = df['Value'].tolist() + label_counts['Sum'].tolist() + city_counts['Sum'].tolist() + region_counts['Sum'].tolist() + [notsag_count]

    fig = px.sunburst(
        ids=ids,
        names=names,
        parents=parents,
        values=values,
        maxdepth=2,
        color=colors,
        branchvalues='total',
        color_discrete_sequence=cc.glasbey_dark
    )
    fig.update_traces(sort=False)
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    fig.write_html('Sunburst_Cities.html')

def plot_clusters_sunburst(df: pd.DataFrame) -> None:
    """Export the clusters' sunburst diagram."""
    print("Generating the clusters' sunburst diagram...")
    fig = px.sunburst(
        df,
        path=['Label', 'City', 'ID/Label'],
        values='Value',
        color='Label',
        maxdepth=2,
        color_discrete_sequence=cc.glasbey_dark
    )
    fig.update_traces(sort=False)
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    fig.write_html('Sunburst_Clusters.html')

def reconsider_noisy_probands():
    unique_labels = sorted(df['Label'].unique())
    unique_labels.remove('N/A')

    new_rows = []
    noisy_ids = df[df['Label'] == 'N/A']['ID'].tolist()
    for noisy_id in tqdm(noisy_ids, "Reconsidering the 'noisy' probands"):
        p_values = []
        kinships_for_means = []
        for label in unique_labels:
            compares = df[df['Label'] == label]['ID'].tolist()
            controls = df[df['Label'] != label]['ID'].tolist()
            compare_kinships = [kinships.loc[noisy_id, compare] for compare in compares]
            control_kinships = [kinships.loc[noisy_id, control] for control in controls]
            _, p_value = mannwhitneyu(compare_kinships, control_kinships, alternative='greater')
            p_values.append(p_value)
            kinships_for_means.append(compare_kinships)
        rejects, _, _, _ = multipletests(p_values, alpha=0.05, method='holm')
        for label, reject, kinships_for_mean in zip(unique_labels, rejects, kinships_for_means):
            if reject:
                new_row = df.loc[df['ID'] == noisy_id].copy()
                new_row['Label'] = label
                new_row['Value'] = statistics.mean(kinships_for_mean)
                new_rows.append(new_row.values[0].tolist())
    df1 = df
    df2 = pd.DataFrame(new_rows, columns = ['ID', 'City', 'Region', 'UMAP 1', 'UMAP 2', 'Label', 'Value'])
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop(df[df['Label'] == 'N/A'].index)
    df['Value'] = df.groupby('ID')['Value'].transform(lambda x: x / x.sum())
    df.sort_values(by=['ID', 'Label'], inplace=True)
    df['Label/ID'] = [f"{row.loc['Label']}/{row.loc['ID']}" for _, row in df.iterrows()]
    df['City/Label'] = [f"{row.loc['City']}/{row.loc['Label']}" for _, row in df.iterrows()]
    
    return df

if __name__ == '__main__':
    individuals = sorted(INDEX.keys())
    idx_map = {individual: index for index, individual in enumerate(individuals)}
    probands = get_probands()
    unique_family_members = get_unique_family_members(probands)

    print("Preparing the kinships…")
    kinships = load_npz('../results/kinships.npz')
    kinships = kinships.toarray()
    kinships = pd.DataFrame(kinships, index=unique_family_members, columns=unique_family_members)
    same_origins = get_same_origins(unique_family_members)
    indices = [index for index, proband in enumerate(unique_family_members) if proband in same_origins]
    X = kinships.to_numpy()
    X = X[indices, :]
    X = X[:, indices]
    print("Done.")

    embedding = UMAP(n_components=20, n_neighbors=10, min_dist=0, random_state=42, verbose=True).fit_transform(X)
    labels = HDBSCAN(min_cluster_size=10).fit_predict(embedding)
    plot = UMAP(random_state=42, verbose=True).fit_transform(embedding)

    print("Populating the new data frame…")
    population = same_origins
    df = pd.DataFrame()
    df['ID'] = population
    city_codes = [CITYCODE[FATHER[proband]] for proband in population]
    cities = [CITY[city_code] for city_code in city_codes]
    df['City'] = cities
    df['Region'] = [REGION[city_code] for city_code in city_codes]
    df['UMAP 1'] = embedding[:, 0]
    df['UMAP 2'] = embedding[:, 1]
    df['Label'] = ['N/A' if label == -1 else '  ' + str(label) if label < 10 else ' ' + str(label) if label < 100 else str(label) for label in labels]
    df['Value'] = 1.0
    print("Done.")

    df = test_significance(df)

    df = reconsider_noisy_probands(df)