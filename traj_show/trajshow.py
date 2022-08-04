import os
import pandas as pd
import numpy as np
import webbrowser as wb
import folium
from folium.plugins import HeatMap, MiniMap, MarkerCluster


# draw a heatmap
def draw_heatmap(map):
    data = (
            np.random.normal(size=(100, 3)) *
            np.array([[1, 1, 1]]) +
            np.array([[30.9, 122.52, 1]])
    ).tolist()
    HeatMap(data).add_to(map)


# add minimap
def draw_minimap(map):
    minimap = MiniMap(toggle_display=True,
                      tile_layer='Stamen Watercolor',
                      position='topleft',
                      width=100,
                      height=100)
    map.add_child(minimap)


def draw_circlemarker(loc, spd, cog, map):
    tip = 'Coordinates:' + str(loc) + "\t" + 'Speed:' + str(spd) + '\t' + 'COG:' + str(cog)
    folium.CircleMarker(
        location=loc,
        radius=3.6,
        color="blueviolet",
        stroke=True,
        fill_color='white',
        fill=True,
        weight=1.5,
        fill_opacity=1.0,
        opacity=1,
        tooltip=tip
    ).add_to(map)


# draw a small information marker on the map
def draw_icon(map, loc):
    mk = folium.features.Marker(loc)
    pp = folium.Popup(str(loc))
    ic = folium.features.Icon(color="blue")
    mk.add_child(ic)
    mk.add_child(pp)
    map.add_child(mk)


# draw a stop marker on the map
def draw_stop_icon(map, loc):
    # mk = folium.features.Marker(loc)
    # pp = folium.Popup(str(loc))
    # ic = folium.features.Icon(color='red', icon='anchor', prefix='fa')
    # mk.add_child(ic)
    # mk.add_child(pp)
    # map.add_child(mk)
    folium.Marker(loc).add_to(map)


def draw_line(map, loc1, loc2):
    kw = {"opacity": 1.0, "weight": 6}
    folium.PolyLine(
        smooth_factor=10,
        locations=[loc1, loc2],
        color="red",
        tooltip="Trajectory",
        **kw,
    ).add_to(map)


def draw_lines(map, coordinates, c):
    folium.PolyLine(
        smooth_factor=0,
        locations=coordinates,
        color=c,
        weight=0.5
    ).add_to(map)


# save the result as HTML to the specified path
def open_html(map, htmlpath):
    map.save(htmlpath)
    search_text = 'cdn.jsdelivr.net'
    replace_text = 'gcore.jsdelivr.net'
    with open(htmlpath, 'r', encoding='UTF-8') as file:
        data = file.read()
        data = data.replace(search_text, replace_text)
    with open(htmlpath, 'w', encoding='UTF-8') as file:
        file.write(data)
    chromepath = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    wb.register('chrome', None, wb.BackgroundBrowser(chromepath))
    wb.get('chrome').open(htmlpath, autoraise=1)


# read .csv file
def read_traj_data(path):
    P = pd.read_csv(path, dtype={'DRLATITUDE': float, 'DRLONGITUDE': float})
    locations_total = P.loc[:, ['DRLATITUDE', 'DRLONGITUDE']].values.tolist()
    speed_total = P.loc[:, ['DRSPEED']].values.tolist()
    cog_total = P.loc[:, ['DRDIRECTION']].values.tolist()
    locations_stay = P.loc[P['STATUS'] == 1, ['DRLATITUDE', 'DRLONGITUDE']].values.tolist()
    lct = [P['DRLATITUDE'].mean(), P['DRLONGITUDE'].mean()]
    return locations_total, speed_total, cog_total, locations_stay, lct


def draw_single_traj(csv_path):
    '''
    draw a single trajectory
    :param data: file path
    :return: null
    '''
    locations, spds, cogs, stays, ct = read_traj_data(csv_path)
    m = folium.Map(ct, zoom_start=15, attr='default')  # 中心区域的确定
    folium.PolyLine(  # polyline方法为将坐标用实线形式连接起来
        locations,  # 将坐标点连接起来
        weight=1.0,  # 线的大小为1
        color='blueviolet',  # 线的颜色
        opacity=0.8,  # 线的透明度
    ).add_to(m)  # 将这条线添加到刚才的区域map内
    num = len(locations)
    for i in range(num):
        draw_circlemarker(locations[i], spds[i], cogs[i], m)
    for i in iter(stays):
        draw_stop_icon(m, i)
    output_path = os.getcwd() + './draw/show.html'
    open_html(m, output_path)


def draw_trajs(file_path):
    '''
    draw multiple trajectories
    :param data: file path
    :return: null
    '''
    map = folium.Map([31.1, 122.5], zoom_start=10, attr='default')  # 中心区域的确定
    draw_minimap(map)
    fls = os.listdir(file_path)
    scatterColors = ['blue', 'red', 'yellow', 'cyan', 'purple', 'orange', 'olive', 'brown', 'black', 'm']
    i = 0
    for x in fls:
        i = i + 1
        colorSytle = scatterColors[i % len(scatterColors)]
        df = pd.read_csv(file_path + "/" + x, encoding="gbk")
        df['DRMMSI'] = df['DRMMSI'].apply(lambda _: str(_))
        df['DRLONGITUDE'] = df['DRLONGITUDE'].map(lambda x: x / 1.0)
        df['DRLATITUDE'] = df['DRLATITUDE'].map(lambda x: x / 1.0)
        for shipmmsi, dt in df.groupby('DRMMSI'):
            if len(dt) > 2:
                dt_copy = dt.copy(deep=True)
                dt_copy.sort_values(by='DRGPSTIME', ascending=True, inplace=True)
                locations = dt_copy.loc[:, ['DRLATITUDE', 'DRLONGITUDE']].values.tolist()
                draw_lines(map, locations, colorSytle)
    output_path = os.getcwd() + './draw/show.html'
    open_html(map, output_path)


if __name__ == '__main__':
    csv_path = r'./data/1.csv'
    draw_single_traj(csv_path)
    # csv_path = r'./data'
    # draw_trajs(csv_path)
