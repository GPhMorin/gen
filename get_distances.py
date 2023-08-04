import pandas as pd
import requests
from tqdm import tqdm

LOCATION_FILE = 'lieux_mariage_definition.csv'
GEOGRAPHY_FILE = 'Territoires_avec_points_centraux.csv'

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

def get_coordinates(filename: str) -> tuple:
    """Converts coordinates from the file into a dictionary of coordinates."""
    data = {}
    df = pd.read_csv(filename, sep=';', encoding='cp1252')
    df.columns = ['cityID', 'city', 'longitude', 'latitude']
    for _, row in df.iterrows():
        longitude = float(row.loc['longitude'].replace(',', '.'))
        latitude = float(row.loc['latitude'].replace(',', '.'))
        data[row.loc['city']] = (longitude, latitude)
    return data

COORDINATES = get_coordinates(GEOGRAPHY_FILE)

def get_distance_from_response(response_data):
    try:
        # Parse the JSON response data
        response_json = response_data.json()

        # Extract the distance value from the response
        distance = response_json["routes"][0]["distance"]

        return distance
    except (KeyError, ValueError):
        return None

def get_distance(coordinates1: tuple, coordinates2: tuple) -> float:
    """Compute the distance between two individuals' weddings."""
    longitude1, latitude1 = coordinates1
    longitude2, latitude2 = coordinates2
    url = f"https://router.project-osrm.org/route/v1/driving/{longitude1},{latitude1};{longitude2},{latitude2}?overview=false"
    try:
        # Send the GET request
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            distance = get_distance_from_response(response)
        else:
            print(f"Error: Unable to fetch data. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    return distance

if __name__ == '__main__':
    cities = sorted(set([CITY[citycode] for citycode in CITY.keys() if REGION[citycode] == "Charlevoix" or REGION[citycode] == "Saguenay-Lac-St-Jean"]))
    print(cities)

    distances = pd.DataFrame(index=cities, columns=cities)
    for city1 in tqdm(cities, "Computing the distances between"):
        for city2 in cities:
            if city1 == city2:
                distances.loc[city1, city2] = 0.0
            elif city1 > city2:
                distances.loc[city1, city2] = distances.loc[city2, city1]
            else:
                coordinates1, coordinates2 = COORDINATES[city1], COORDINATES[city2]
                distances.loc[city1, city2] = get_distance(coordinates1, coordinates2) / 1000.
            
    distances.to_csv("distances.csv")