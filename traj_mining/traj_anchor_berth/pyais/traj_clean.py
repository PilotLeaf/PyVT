import pandas as pd
import numpy as np
from vincenty import vincenty
import warnings

warnings.filterwarnings("ignore")


def heuristic_clean(data, vthreshold: float):
    '''
    :param data: raw trajectory
    :param vthreshold: speed threshold
    :return: cleaned trajectory
    '''
    tdf = pd.DataFrame(columns=data.columns, index=data.index)
    tdf.drop(tdf.index, inplace=True)

    data.drop_duplicates()
    data = data.reset_index(drop=True)
    for shipmmsi, dt in data.groupby('DRMMSI'):
        if len(str(shipmmsi)) > 5:
            dt_copy = dt.copy(deep=True)
            dt_copy.sort_values(by='DRGPSTIME', ascending=True, inplace=True)
            dt_copy = dt_copy.reset_index(drop=True)
            dt_copy['NOISE'] = 0
            for idx, di in dt_copy.iterrows():
                if di['NOISE'] == 0:
                    if idx < len(dt_copy) - 1:
                        t1 = di['DRGPSTIME']
                        t2 = dt_copy.loc[idx + 1, 'DRGPSTIME']
                        Δt = (t2 - t1) / 3600.0
                        pt1 = (di['DRLATITUDE'], di['DRLONGITUDE'])
                        pt2 = (dt_copy.loc[idx + 1, 'DRLATITUDE'], dt_copy.loc[idx + 1, 'DRLONGITUDE'])
                        Δd = vincenty(pt1, pt2) * 1000 / 1852.25
                        Δv = Δd / Δt
                        if Δv > vthreshold:
                            dt_copy.loc[idx + 1, 'NOISE'] = 1
                            print("1")
                else:
                    continue
            dt_copy_new = dt_copy.query('NOISE==0')
            dt_copy_new = dt_copy_new.drop(['NOISE'], axis=1)
            tdf = tdf.append(dt_copy_new, ignore_index=True)
    return tdf


def sw_clean(data, sw: int):
    '''
    :param data: raw trajectory
    :param sw: the size of sliding window
    :return: cleaned trajectory
    '''
    tdf = pd.DataFrame(columns=data.columns, index=data.index)
    tdf.drop(tdf.index, inplace=True)

    data.drop_duplicates()
    data = data.reset_index(drop=True)
    for shipmmsi, dt in data.groupby('DRMMSI'):
        if len(str(shipmmsi)) > 6:
            dt_copy = dt.copy(deep=True)
            dt_copy.sort_values(by='DRGPSTIME', ascending=True, inplace=True)
            dt_copy = dt_copy.reset_index(drop=True)
            dt_copy['NOISE'] = 0
            copylength = len(dt_copy)
            num_samples = copylength // sw + 1
            for idx in np.arange(num_samples):
                start_x = sw * idx
                end_x = start_x + sw - 1
                end_x = (copylength - 1) if end_x > (copylength - 1) else end_x
                dt_temp = dt_copy.loc[start_x:end_x]
                lats = dt_temp.loc[:, "DRLATITUDE"]
                longs = dt_temp.loc[:, "DRLONGITUDE"]
                stdlat = lats.std()
                meanlat = lats.mean()
                stdlog = longs.std()
                meanlog = longs.mean()
                for jdx, di in dt_temp.iterrows():
                    if abs(di['DRLATITUDE'] - meanlat) > abs(1.5 * stdlat) or abs(di['DRLONGITUDE'] - meanlog) > abs(
                            1.5 * stdlog):
                        dt_copy.loc[jdx, 'NOISE'] = 1
                        print("1")
                    pass

            dt_copy_new = dt_copy.query('NOISE==0')
            dt_copy_new = dt_copy_new.drop(['NOISE'], axis=1)
            tdf = tdf.append(dt_copy_new, ignore_index=True)
    return tdf
