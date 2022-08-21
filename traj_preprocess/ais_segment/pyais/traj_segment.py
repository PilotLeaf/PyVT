import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def segment(df, tsinterval):
    """
    Trajectory segmentation.
    :param df: an array
    :type df: DataFrame
    :param tsepsilon: time threshold
    :type tsepsilon: float
    :param tdf: return segmented trajectory
    :type tdf: DataFrame
    """
    tdf = pd.DataFrame(columns=df.columns, index=df.index)
    tdf.drop(tdf.index, inplace=True)
    tdf['tdiff'] = 0

    for shipmmsi, dt in df.groupby('DRMMSI'):
        data = dt.copy(deep=True)
        data.sort_values(by='DRGPSTIME', ascending=True, inplace=True)
        data = data.reset_index(drop=True)
        data['tdiff'] = data['DRGPSTIME'].diff().fillna(0)
        i = 0
        lastindex = 0
        for idx, di in data.iterrows():
            if di['tdiff'] > tsinterval and idx >= 1:
                data.loc[lastindex:idx - 1, 'DRMMSI'] = str(di['DRMMSI']) + '_' + str(i)
                i = i + 1
                lastindex = idx
        tdf = tdf.append(data, ignore_index=True)
    tdf = tdf.drop(['tdiff'], axis=1)
    return tdf