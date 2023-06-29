from scipy.sparse import load_npz
import numpy as np
import n2d
import pandas as pd
import plotly.express as px
import colorcet as cc
import itertools
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from umap import UMAP
from hdbscan import HDBSCAN

city_names = {}
df = pd.read_csv('../data/lieux_mariage_definition.csv', usecols=[0, 1, 2], encoding='cp1252')
df.columns = ['name', 'code', 'region']
for index, row in df.iterrows():
    if row.loc['code'] == 'UrbIdMariage':
        continue
    city_names[int(row.loc['code'])] = row.loc['name']

regions = {}
for index, row in df.iterrows():
    if row.loc['code'] == 'UrbIdMariage':
        continue
    regions[int(row.loc['code'])] = row.loc['region']
regions[16674] = 'Nouveau-Brunswick'
regions[20228] = 'Nouveau-Brunswick'
regions[16915] = 'Ontario'

def get_dict(filename: str) -> dict:
    """Converts lines from the file into a dictionary of parents and indices."""
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
    data = {}
    for index, line in enumerate(lines):
        child, father, mother, _ = line.strip().split('\t')
        data[int(child)] = (int(father), int(mother), index)
    return data

def get_probands(data: dict) -> list:
        """Extracts the probands as an ordered list."""
        probands = set(data.keys()) - {parent for individual in data.keys() for parent in data[individual][:2] if parent}
        return sorted(list(probands))

def get_unique_family_members(data: dict, probands: list) -> list:
        """Limits families to one member each."""
        visited_parents = set()
        unique_family_members = []
        for proband in probands:
            father, mother, _ = data[proband]
            if (father, mother) not in visited_parents:
                unique_family_members.append(proband)
                visited_parents.add((father, mother))
        return unique_family_members

def get_city(filename: str) -> dict:
    """Converts cities from the file into a dictionary of cities and IDs."""
    data = {}
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
        for line in lines:
            line = line.strip().split('\t')
            data[int(line[0])] = int(line[4])
    return data

def get_weddate(filename: str) -> dict:
    """Converts wedding years from the file into a dictionary of wedding years and IDs."""
    data = {}
    with open(filename, 'r') as infile:
        lines = infile.readlines()[1:]
        for line in lines:
            line = line.strip().split('\t')
            data[int(line[0])] = int(line[3])
    return data
        
def get_same_origins(data: dict, probands: list) -> list:
    """Limits probands to those who married at the same place as their parents."""
    same_origins = []
    city = get_city('../data/tous_individus_dates_locations.txt')
    for proband in probands:
        try:
            father, mother, _ = data[proband]
            p_grandfather, p_grandmother, _ = data[father]
            m_grandfather, m_grandmother, _ = data[mother]
            if city[p_grandfather] == city[p_grandmother] == city[m_grandfather] == city[m_grandmother] == city[father] == city[mother] != 0:
                same_origins.append(proband)
        except KeyError:
            continue
    return same_origins

def get_charlevoix_saglac(data: dict, probands: list) -> list:
    """Filter out probands whose origins are from outside Charlevoix and Saguenay--Lac-Saint-Jean."""
    charlevoix_saglac = []
    city = get_city('../data/tous_individus_dates_locations.txt')
    for proband in probands:
        parent, _, _ = data[proband]
        grandparent, _, _ = data[parent]
        if regions[city[grandparent]] == "Charlevoix" or regions[city[grandparent]] == "Saguenay-Lac-St-Jean":
            charlevoix_saglac.append(proband)
    return charlevoix_saglac

data = get_dict('../data/tous_individus_pro1931-60_SAG.asc')
probands = get_probands(data)
unique_family_members = get_unique_family_members(data, probands)
same_origins = get_same_origins(data, unique_family_members)
# charlevoix_saglac = get_charlevoix_saglac(data, same_origins)
population = same_origins
indices = [index for index, proband in enumerate(unique_family_members) if proband in population]
city = get_city('../data/tous_individus_dates_locations.txt')
cities = [city[data[data[individual][0]][0]] for individual in population]
weddate = get_weddate('../data/tous_individus_dates_locations.txt')

print("WEDDING YEARS")
print(f"Probands: {np.array([weddate[proband] for proband in population]).mean()}")
parents = []
grandparents = []
for proband in population:
    father, mother, _ = data[proband]
    parents.extend([father, mother])
    if father in data:
        father_grandparents = data[father][:2]
        grandparents.extend([grandparent for grandparent in father_grandparents if grandparent])
    if mother in data:
        mother_grandparents = data[mother][:2]
        grandparents.extend([grandparent for grandparent in mother_grandparents if grandparent])
print(f"Parents: {np.array([weddate[parent] for parent in parents]).mean()}")
print(f"Grandparents: {np.array([weddate[grandparent] for grandparent in grandparents]).mean()}")

print("Loading the matrix...")
matrix = load_npz('../results/kinships.npz')

print("Transforming the sparse matrix to an array...")
X = matrix.toarray()
kinship_df = pd.DataFrame(X, index=unique_family_members, columns=unique_family_members)

print("Preparing the labels")
y = np.array(cities)

X = X[indices, :]
X = X[:, indices]

print(X.shape)
print(len(y))
print(len(set(y)))

print("Encoding the data...")
"""
latent_dims = 50
ae = n2d.AutoEncoder(X.shape[-1], latent_dims)
cls = n2d.manifold_cluster_generator(UMAP, {"n_neighbors": 10, "min_dist": 0, "random_state": 42}, HDBSCAN, {"min_cluster_size": 15})
cluster = n2d.n2d(ae, cls)
labels = cluster.fit_predict(X, patience=100)
plot = cluster.hle
"""
latent_dims = 20
umap50 = UMAP(
    n_components=latent_dims,
    n_neighbors=10,
    min_dist=0,
    random_state=42,
    verbose=True
    )
latent_emb = umap50.fit_transform(X)
scan = HDBSCAN(min_cluster_size=10)
labels = scan.fit_predict(latent_emb)
umap2 = UMAP(
    random_state=42,
    verbose=True
)
plot = umap2.fit_transform(X)

print("Generating the scatter plot...")
umap_df = pd.DataFrame()
umap_df['ID'] = population
umap_df['City'] = [city_names[city] for city in y]
umap_df['Region'] = [regions[city] for city in y]
umap_df['UMAP 1'] = plot[:, 0]
umap_df['UMAP 2'] = plot[:, 1]
umap_df['Label'] = ['N/A' if label == -1 else '  ' + str(label) if label < 10 else ' ' + str(label) if label < 100 else str(label) for label in labels]
umap_df['Label/ID'] = [f"{row.loc['Label']}/{row.loc['ID']}" for _, row in umap_df.iterrows()]
umap_df['Value'] = np.ones(len(umap_df))

saglac_cities = umap_df[umap_df['Region'] == 'Saguenay-Lac-St-Jean']['City'].unique()
location_dict = dict(zip(saglac_cities, saglac_cities))
location_dict.update({region: region for region in umap_df['Region'].unique()})
umap_df['Location'] = umap_df['City'].map(location_dict).fillna(umap_df['Region'].map(location_dict))

sector = ["Saguenay-Lac-St-Jean" if row.loc['Region'] == "Saguenay-Lac-St-Jean"
          else "Rest of Canada" for _, row in umap_df.iterrows()]
umap_df['Sector'] = sector

umap_df.sort_values(by=['Label', 'ID'], inplace=True)

clusters = list(umap_df['Label'].unique())
number_before = len(clusters)

# Store the rows in a list
rows = []

# Group by cluster
grouped = umap_df.groupby('Label')

for cluster, group in tqdm(grouped, desc="Preparing the box plot"):
    individuals = group['ID'].tolist()
    intracluster_combinations = itertools.combinations(individuals, 2)
    
    # Iterate over combinations and append to list
    for individual1, individual2 in intracluster_combinations:
        kinship = kinship_df.loc[individual1, individual2]
        rows.append({"Cluster": f"{cluster} (n={group.size})", "Kinship": kinship})

# Create the DataFrame once, outside the loop
boxplots = pd.DataFrame(rows)

# Create and save the plot
fig = px.box(boxplots, x='Cluster', y='Kinship')
fig.write_html("N2D-boxplot.html")

print("Calculating if the clusters are statistically significant...")
if 'N/A' in clusters:
    clusters.remove('N/A')
tests = []
for cluster1 in tqdm(clusters, desc="Grouping the cluster combinations"):
    for cluster2 in clusters:
        if cluster1 >= cluster2:
            continue
        individuals1 = umap_df[umap_df['Label'] == cluster1]['ID'].tolist()
        individuals2 = umap_df[umap_df['Label'] == cluster2]['ID'].tolist()
        intracluster_combinationsA = itertools.combinations(individuals1, 2)
        intracluster_combinationsB = itertools.combinations(individuals2, 2)
        intercluster_combinations = itertools.product(individuals1, individuals2)
        intracluster_kinshipsA = [kinship_df.loc[individual1, individual2]
                                 for individual1, individual2 in intracluster_combinationsA]
        intracluster_kinshipsB = [kinship_df.loc[individual1, individual2]
                                  for individual1, individual2 in intracluster_combinationsB]
        intercluster_kinships = [kinship_df.loc[individual1, individual2]
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

for label in umap_df['Label'].unique():
    if label not in clusters_map:
        clusters_map[label] = label
        
umap_df['Label'] = umap_df['Label'].map(clusters_map)

number_after = len(umap_df['Label'].unique())

if number_after == number_before:
    print("All clusters are statistically significant!")
else:
    print(f"{number_after} clusters are statistically significant.")

umap_df.sort_values(by=['Label', 'ID'], inplace=True)
umap_df['Label/ID'] = [f"{row.loc['Label']}/{row.loc['ID']}" for _, row in umap_df.iterrows()]

print("Generating the clusters' scatter plot...")
fig = px.scatter(
    umap_df.sort_values(by='Label'),
    x='UMAP 1',
    y='UMAP 2',
    color='Label',
    hover_name='ID',
    hover_data=['City', 'Region'],
    template='simple_white',
    color_discrete_sequence=cc.glasbey_dark
)
fig.write_html('N2D_Scatter_Clusters.html')

print("Generating the cities' sunburst diagram...")
ids = []
names = []
parents = []
for _, row in umap_df.iterrows():
    proband = row['Label/ID']
    label = row['Label']
    city = row['City']
    region = row['Region']
    ids.append(proband)
    names.append(label)
    parents.append(city)
for city in sorted(umap_df['City'].unique()):
    region = umap_df.loc[umap_df['City'] == city, 'Region'].iloc[0]
    ids.append(city)
    names.append(city)
    parents.append(region)
for region in sorted(umap_df['Region'].unique()):
    ids.append(region)
    names.append(region)
    parents.append("")
    """
    if region == "Saguenay-Lac-St-Jean":
        parents.append("")
    else:
        parents.append("Rest of Canada")
    """
# ids.append("Rest of Canada")
# names.append("Rest of Canada")
# parents.append("")

city_modes = umap_df.sort_values(by='City').groupby('City')['Label'].apply(lambda x: x.mode()[0]).reset_index()
city_counts = umap_df.sort_values(by='City').groupby('City')['Label'].size().reset_index(name='Count')
region_modes = umap_df.sort_values(by='City').groupby('Region')['Label'].apply(lambda x: x.mode()[0]).reset_index()
region_counts = umap_df.sort_values(by='City').groupby('Region')['Label'].size().reset_index(name='Count')
# notsag_count = umap_df[umap_df['Region'] != "Saguenay-Lac-St-Jean"]['ID'].count()
# notsag_mode = umap_df[umap_df['Region'] != "Saguenay-Lac-St-Jean"]['Label'].mode()[0]
colors = umap_df['Label'].tolist() + city_modes['Label'].tolist() + region_modes['Label'].tolist() # + [notsag_mode]
values = list(np.ones(len(umap_df), dtype=int)) + city_counts['Count'].tolist() + region_counts['Count'].tolist() # + [notsag_count]

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
fig.write_html('N2D_Sunburst_Cities.html')

print("Generating the clusters' sunburst diagram...")
fig = px.sunburst(
    umap_df,
    path=['Label', 'City', 'Label/ID'],
    color='Label',
    maxdepth=2,
    color_discrete_sequence=cc.glasbey_dark
)
fig.update_traces(sort=False)
fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
fig.write_html('N2D_Sunburst_Clusters.html')

print("Saving the new data...")
umap_df.to_csv("grands-n2d.csv", index=False)