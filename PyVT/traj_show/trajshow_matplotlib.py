import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(file_path):
    """
    import trajectory data
    :return: trajectory data
    """
    data = pd.read_csv(file_path, usecols=['DRGPSTIME', 'DRMMSI', 'DRLATITUDE', 'DRLONGITUDE'])
    data.rename(columns={'DRLONGITUDE': 'long', 'DRLATITUDE': 'lat', 'DRGPSTIME': 't', 'DRMMSI': 'mmsi'}, inplace=True)
    data['long'] = data['long'].map(lambda x: x / 600000.0)
    data['lat'] = data['lat'].map(lambda x: x / 600000.0)
    return data


if __name__ == '__main__':
    trajectories = load_data('./data/1.csv')
    params = {'axes.titlesize': 'large',
              'legend.fontsize': 14,
              'legend.handlelength': 3}
    plt.rcParams.update(params)

    for shipmmsi, dt in trajectories.groupby('mmsi'):
        plt.plot(dt['long'].values, dt['lat'].values, color='green', linewidth=0.5)

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title('Preprocessed Trajectories', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
