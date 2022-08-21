import pandas as pd
import numpy as np
import time
from decimal import Decimal
import matplotlib.pyplot as plt
from pyais import traj_kalmanfilter

if __name__ == '__main__':
    df = pd.read_csv("../data/1.csv")
    ais_origin = []
    ais_data = []
    ais_kalman = traj_kalmanfilter.KalmanFilter()

    for index, row in df.iterrows():
        spd = Decimal(row['SPEED'] * 1852.25 / 3600)
        tm = str(row['Date']) + ' ' + str(row['Time'])
        timeArray = time.strptime(tm, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        ais_origin.append([row['Longitude'], row['Latitude']])
        coords = str(row['Longitude']) + ',' + str(row['Latitude'])
        pt = ais_kalman.process(speed=spd, coordinate=coords, time_stamp=timeStamp, accuracy=10.0)
        ais_data.append(ais_kalman._split_coordinate(pt))
    res = list(filter(None, ais_data))
    x, y = np.array(res).T
    reso = list(filter(None, ais_origin))
    xo, yo = np.array(reso).T
    plt.plot(xo, yo, color='b', marker='*', linestyle='--', linewidth=0.5, label='Raw trajectory')
    plt.plot(x, y, color='r', marker='o', ms=5, linestyle='-', linewidth=0.5, label='Filtered trajectory')
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.title('Kalman Filter', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
