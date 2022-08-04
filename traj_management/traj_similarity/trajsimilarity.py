from dtaidistance import dtw
from dtaidistance import dtw_ndim
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from haversine.haversine import haversine


def load_data(file_path):
    """
    import trajectory data
    :return: trajectory data
    """
    data = pd.read_csv(file_path, usecols=['DRGPSTIME', 'DRMMSI', 'DRLATITUDE', 'DRLONGITUDE'])
    data.rename(columns={'DRLONGITUDE': 'long', 'DRLATITUDE': 'lat', 'DRGPSTIME': 't', 'DRMMSI': 'mmsi'}, inplace=True)
    trajectories = []
    grouped = data[:].groupby('mmsi')
    for name, group in grouped:
        if len(group) > 1:
            group = group.sort_values(by='t', ascending=True)
            group['long'] = group['long'].apply(lambda x: x * (1 / 1.0))  # 以°为单位
            group['lat'] = group['lat'].apply(lambda x: x * (1 / 1.0))
            loc = group[['lat', 'long']].values  # 原始
            trajectories.append(loc)
    return trajectories


def dist_computer(trajectories):
    distance_matrix = dtw_ndim.distance_matrix_fast(trajectories, 2)
    distance_matrix = np.array(distance_matrix)
    np.savetxt('1.txt', distance_matrix, fmt='%.5f')
    return distance_matrix


def dist_computer_fastdtw(trajectories):
    distance_matrix = []
    i = 0
    l = len(trajectories)
    for traj in trajectories:
        i = i + 1
        print(str(i) + "/" + str(l))
        v = []
        for q in trajectories:
            distance, path = fastdtw(traj, q, dist=haversine)
            v.append(distance)
        distance_matrix.append(v)
    distance_matrix = np.array(distance_matrix)
    np.savetxt('1.txt', distance_matrix, fmt='%.5f')
    return distance_matrix


if __name__ == '__main__':
    trajectories = load_data('./data/1.csv')
    dist_matrix = dist_computer(trajectories)
