from dtaidistance import dtw_ndim
from dtaidistance import dtw
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hdbscan


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
            group['long'] = group['long'].apply(lambda x: x * (1 / 600000.0))  # 以°为单位
            group['lat'] = group['lat'].apply(lambda x: x * (1 / 600000.0))
            loc = group[['lat', 'long']].values  # 原始
            trajectories.append(loc)
    return trajectories


def dist_computer(trajectories):
    distance_matrix = dtw_ndim.distance_matrix_fast(trajectories, 2)
    distance_matrix = np.array(distance_matrix)
    return distance_matrix


if __name__ == '__main__':
    trajectories = load_data('../data/final1.csv')
    dist_matrix = dist_computer(trajectories)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, cluster_selection_epsilon=0, metric='precomputed')
    clusterer.fit(dist_matrix)
    lbs = clusterer.labels_.copy()
    li = list(set(lbs))

    params = {'axes.titlesize': 'large',
              'legend.fontsize': 14,
              'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.style.available
    # plt.style.use("dark_background")

    cls = ['red', 'green', 'brown', 'pink', 'yellow', 'black', 'rosybrown', 'royalblue', 'purple', 'tomato',
           'cyan', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'lightcyan', 'lavender',
           'lavenderblush', 'mediumseagreen', 'mediumslateblue', 'teal', 'mediumspringgreen', 'mediumturquoise',
           'mediumvioletred', 'maroon', 'yellowgreen',
           'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
           'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'darkturquoise', 'darkviolet', 'deeppink',
           'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
           'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred',
           'indigo', 'ivory', 'khaki', 'lemonchiffon', 'lightblue', 'aliceblue', 'lightcoral', 'lightgoldenrodyellow',
           'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
           'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'paleturquoise',
           'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'plum', 'powderblue', 'saddlebrown', 'salmon',
           'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray',
           'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'violet', 'wheat', 'white', 'whitesmoke',
           'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'turquoise',
           'lawngreen', 'blue', 'blueviolet', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
           'cornflowerblue', 'cornsilk', 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
           'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
           'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray']
    for index, value in enumerate(lbs):
        if value >= 0:
            t = trajectories[index]
            x, y = t.T
            if len(t) > 0:
                plt.plot(y, x, color=cls[value], alpha=1.0, lw=0.4)
        else:
            t = trajectories[index]
            x, y = t.T
            plt.plot(y, x, color='red', alpha=0.0, lw=0.4)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title('Trajectory Clustering', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
