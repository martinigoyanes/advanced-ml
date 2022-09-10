import requests
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

distances = {
    "Stockholm": {}, "Budapest": {}, "Oslo": {}, "Copenhagen": {}, "Helsinki": {},
    "Berlin": {}, "London": {}, "Amsterdam": {},
    "Vienna": {}, "Warsawa": {}, "Madrid": {}, "Paris": {}, "Rome": {}, "Moscow": {},
    "Dublin": {}, "Athens": {}, "Zurich": {}, "Prague": {}, "Reykjavik": {},
    "Kiev": {}, "Bucharest": {}, "Minsk": {}, "Monaco": {}, "San Marino": {}, "Sarajevo": {},
    "Sofia": {}, "Zagreb": {}, "Belgrade": {},
    "Andorra la Vella": {}, "Gothenburg": {}, "Birmingham": {}, "Manchester": {},
    "Liverpool": {}, "Barcelona": {}, "Amman": {},
    "Ashgabat": {}, "Dili": {}, "Jerusalem": {}, "Dushanbe": {}, "Doha": {},
    "Lhasa": {}, "Baghdad": {}, "Beirut": {}, "Baku": {}, "Colombo": {},
    "Islamabad": {}, "Singapore": {}, "Tokyo": {}, "Kabul": {}, "Beijing": {},
    "Damascus": {}, "Seoul": {}, "New Delhi": {}, "Ankara": {},
    "Bangkok": {}, "Hanoi": {}, "Riyadh": {}, "Abu Dhabi": {}, "Kuwait City": {},
    "Dhaka": {}, "Osaka": {}, "Tehran": {}, "Ulaanbaatar": {},
    "Taipei": {}, "Jakarta": {}, "Manila": {}, "Washington": {}, "Ottawa": {},
    "Mexico City": {}, "Belmopan": {}, "Guatemala City": {},
    "Tegucigalpa": {}, "San Salvador": {}, "Managua": {}, "San Jose": {},
    "Panama City": {}, "St. John's": {}, "Nassau": {}, "Bridgetown": {},
    "Havana": {}, "Roseau": {}, "Santo Domingo": {}, "St Georges": {},
    "Port-au-Prince": {}, "Kingston": {}, "Basseterre": {}, "Castries": {},
    "Kingstown": {}, "Buenos Aires": {}, "Sucre": {}, "Brasilia": {},
    "Santiago": {}, "Bogota": {}, "Quito": {}, "Georgetown": {}, "Asuncion": {},
    "Lima": {}, "Paramaribo": {}, "Montevideo": {}, "Caracas": {},
    "WashingtonDC": {}, "NewYork": {}, "Montreal": {}, "Chicago": {}, "Atlanta": {},
    "Dallas": {}, "Austin": {}, "Denver": {}, "Seattle": {}, "Las Vegas": {},
    "Edmonton": {}, "Casablanca": {}, "Tunisia": {}, "Tamanrasset": {},
    "El Cairo": {}, "Agadez": {}, "Dakar": {}, "Duala": {}, "Gitega": {},
    "Nairobi": {}, "Lusaka": {}, "Johannesburgo": {}
}


def get_distances():
    endpoint = "https://www.distance24.org/route.json?stops="

    for start_city in distances.keys():
        print(f"Calculating distances for {start_city}")
        for end_city in distances.keys():
            if start_city not in distances[end_city].keys():
                cities = f"{start_city}|{end_city}"
                url = f"{endpoint}{cities}"
                response = requests.get(url).json()
                dist = response["distance"]
            else:
                dist = distances[end_city][start_city]

            distances[start_city][end_city] = dist
            print(f"{start_city} -> {end_city}: {dist}")

    print(distances)

    distList = []

    for start_city in distances.keys():
        dists = []
        for d in distances[start_city].values():
            dists.append(d)
        distList.append(dists)

    distMatrix = np.array(distList)

    print("\n\n\t##Matrix##")
    print(distMatrix)

    with open('distances.txt', 'w') as outfile:
        json.dump(distances, outfile)


def classical_mds():
    '''
    Apply  classical  MDS  to  compute  an  (x,  y)  coordinate  
    for  each  city  in  yourdataset, 
    given the distance matrix D. Plot the cities on a plane 
    using the coor-dinates you computed.  
    You may want to annotate the cities by their name, 
    orabbreviation, use different colors 
    to indicate continents, or regions, etc.  
    Discusshow good is the reconstructed map your 
    created using classical MDS. For thistask, you should 
    implement MDS by yourself, by relying only 
    on a package foreigenvector decomposition, that is, do not 
    try to find a library that implements MDS
    '''
    # Loading downloaded distances
    with open('distances.txt') as json_file:
        distances_json = json.load(json_file)

    # Converting distances to numpy matrix
    distList = []

    for start_city in distances_json.keys():
        dists = []
        for d in distances_json[start_city].values():
            dists.append(d)
        distList.append(dists)

    distMatrix = np.array(distList)

    print(
        f"\t## Distance Matrix ##\t({distMatrix.shape[0]}x{distMatrix.shape[1]})")
    print(distMatrix)

    distMatrix = distMatrix**2
    print(
        f"\t## Distance Matrix ##\t({distMatrix.shape[0]}x{distMatrix.shape[1]})")
    print(distMatrix)

    N = distMatrix.shape[0]
    K = 2  # Since we want to plot (x, y) coordingates, i.e, 2-dimesnions

    # Compute Centering Matrix for double centering
    centeringMatrix = np.identity(N) - (np.ones((N, 1)) @ np.ones((1, N)))/N

    # Compute Gram matrix with double centering from distance matrix
    gramMatrix = -(1/2)*(centeringMatrix @ distMatrix @ centeringMatrix)

    # Eigen-decomposition
    w, v = np.linalg.eigh(gramMatrix)
    # The values are returned in ascending order, and we want the K greatest so take K last elemnents
    w = w[-K:]
    v_T = v.T[-K:]

    assert np.all(w > 0)
    latentMatrix = np.diag(np.sqrt(w)) @ v_T

    print(
        f"\t## Latent Matrix ##\t({latentMatrix.shape[0]}x{latentMatrix.shape[1]})")
    print(latentMatrix)

    plt.figure(figsize=(20, 20))

    plt.scatter(latentMatrix[0, :], latentMatrix[1, :])
    labels = [city[:8] for city in distances_json.keys()]
    for (index, label), x, y in zip(enumerate(labels), latentMatrix[0, :], latentMatrix[1, :]):
        if index % 2 == 0:
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(-10, 10),
                textcoords='offset points'
            )
    plt.savefig('classical-map.png')


def metric_mds():
    '''
    Repeat the task of 1.4.12, but using metric MDS this time.  
    You may tune theparameters of the method until you are
    satisfied with the results.  
    Discuss theparameters that you tuned and their effect 
    on the end result.  
    Discuss the qualityof your map, and compare it with the 
    one you obtained by classical MDS. 
    For this task, you are free to search for an implementation 
    of metric MDS and useit as a black box.
    '''
    # Loading downloaded distances
    with open('distances.txt') as json_file:
        distances_json = json.load(json_file)

    # Converting distances to numpy matrix
    distList = []

    for start_city in distances_json.keys():
        dists = []
        for d in distances_json[start_city].values():
            dists.append(d)
        distList.append(dists)

    distMatrix = np.array(distList)

    mds = MDS(n_components=2, metric=True, n_init=16,
              verbose=3, dissimilarity="precomputed")
    latentMatrix = mds.fit_transform(distMatrix)

    print(
        f"\t## Latent Matrix ##\t({latentMatrix.shape[0]}x{latentMatrix.shape[1]})")
    print(latentMatrix)

    plt.figure(figsize=(20, 20))

    plt.scatter(latentMatrix[:, 0], latentMatrix[:, 1])
    labels = [city[:8] for city in distances_json.keys()]
    for (index, label), x, y in zip(enumerate(labels), latentMatrix[:, 0], latentMatrix[:, 1]):
        if index % 2 == 0:
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(-10, 10),
                textcoords='offset points'
            )
    plt.savefig('metric-map.png')

    print(mds.get_params())


def main():
    # get_distances()
    classical_mds()
    metric_mds()


main()
