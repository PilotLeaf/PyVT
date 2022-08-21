import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from typing import Dict, List, Union, Optional
from vincenty import vincenty

# types
TrajData = Dict[str, Union[List[float]]]


def traj_load(traj_file: str) -> TrajData:
    trajs_data = {'lat': [], 'lon': [], 'tstamp': [], 'speed': []}
    df = pd.read_csv(traj_file)
    dt_c = df.copy(deep=True)
    dt_c['DRLONGITUDE'] = dt_c['DRLONGITUDE'].map(lambda x: x / 1.0)
    dt_c['DRLATITUDE'] = dt_c['DRLATITUDE'].map(lambda x: x / 1.0)
    dt_c['DRSPEED'] = dt_c['DRSPEED'].map(lambda x: x / 1.0)
    for index, row in dt_c.iterrows():
        trajs_data['lat'].append(row['DRLATITUDE'])
        trajs_data['lon'].append(row['DRLONGITUDE'])
        trajs_data['speed'].append(row['DRSPEED'])
        trajs_data['tstamp'].append(row['DRGPSTIME'])
    return trajs_data


def traj_calculate_distance(traj_data: TrajData) -> List[float]:
    traj_dist = np.zeros(len(traj_data['lat']))
    for i in range(len(traj_dist) - 1):
        lat1 = traj_data['lat'][i]
        lon1 = traj_data['lon'][i]
        lat2 = traj_data['lat'][i + 1]
        lon2 = traj_data['lon'][i + 1]
        pts = (lat1, lon1)
        pte = (lat2, lon2)
        s = vincenty(pts, pte) * 1000  # unit meter
        traj_dist[i + 1] = s
    return traj_dist.tolist()


def traj_interpolate(traj_file: str, res: float = 1.0, num: Optional[int] = None) -> TrajData:
    '''
    :param traj_data: raw trajectory filename
    :param res: time resolution
    :param num: None
    :return: interpolated trajectory
    '''
    traj_data = traj_load(traj_file)
    if res <= 0.0:
        raise ValueError('res must be > 0.0')
    if num is not None and num < 0:
        raise ValueError('num must be >= 0')
    _traj_dist = traj_calculate_distance(traj_data)
    xi = np.cumsum(_traj_dist)
    yi = np.array([traj_data[i] for i in ('lat', 'lon', 'speed', 'tstamp') if traj_data[i]])
    num = num if num is not None else int(np.ceil(xi[-1] / res))
    x = np.linspace(xi[0], xi[-1], num=num, endpoint=True)
    y = pchip_interpolate(xi, yi, x, axis=1)
    traj_data_interp = {'lat': list(y[0, :]), 'lon': list(y[1, :]), 'speed': list(y[2, :]), 'tstamp': list(y[-1, :])}
    return traj_data_interp


def traj_calculate_distance_ts(traj_data: TrajData) -> List[float]:
    traj_dist = np.zeros(len(traj_data['lat']))
    for i in range(len(traj_dist) - 1):
        s = int(traj_data['tstamp'][i + 1]) - int(traj_data['tstamp'][i])
        traj_dist[i + 1] = s
    return traj_dist.tolist()


def traj_interpolate_df(traj_data, res: float = 1.0, num: Optional[int] = None) -> TrajData:
    '''
    :param traj_data: raw trajectory dataframe
    :param res: time resolution
    :param num: None
    :return: interpolated trajectory
    '''
    if res <= 0.0:
        raise ValueError('res must be > 0.0')
    if num is not None and num < 0:
        raise ValueError('num must be >= 0')
    _traj_dist = traj_calculate_distance_ts(traj_data)
    xi = np.cumsum(_traj_dist)
    yi = np.array([traj_data[i] for i in ('lat', 'lon', 'speed', 'tstamp') if traj_data[i]])
    num = num if num is not None else int(np.ceil(xi[-1] / res))
    x = np.linspace(xi[0], xi[-1], num=num, endpoint=True)
    y = pchip_interpolate(xi, yi, x, axis=1)
    return y.T
