from dataclasses import dataclass
from queue import PriorityQueue
import pandas as pd
import numpy as np


def degree_to_radian(deg):
    return deg * np.pi / 180.0


def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6_378_137.0

    lat1_rad = degree_to_radian(lat1)
    lon1_rad = degree_to_radian(lon1)
    lat2_rad = degree_to_radian(lat2)
    lon2_rad = degree_to_radian(lon2)

    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad

    a = np.sin(d_lat / 2) ** 2 + np.sin(d_lon / 2) ** 2 * np.cos(lat1_rad) * np.cos(lat2_rad)
    return radius * 2 * np.arcsin(np.sqrt(a))


@dataclass
class Neighbour:
    key: np.int64  # id
    length: np.float64  # street length
    lat: np.float64
    lon: np.float64

    def get_distance(self, lat, lon):
        return haversine_distance(lat, lon, self.lat, self.lon)


def get_neighbours(node, graph: pd.DataFrame):
    res = []
    rows = graph[(graph['u'] == node) | ((graph['v'] == node) & (graph['oneway'] == 'no'))].iterrows()
    for _, row in rows:
        u = 'u'
        if row['u'] == node:
            u = 'v'

        res.append(Neighbour(key=row[u], length=row['length'], lat=row[f"{u}_lat"], lon=row[f"{u}_lon"]))
    return res


def get_node_id(lat, lon, graph: pd.DataFrame):
    tmp = graph[(graph['v_lat'] == lat) & (graph['v_lon'] == lon)]
    if len(tmp) > 0:
        return tmp.iloc[0]['v']

    tmp = graph[(graph['u_lat'] == lat) & (graph['u_lon'] == lon)]
    if len(tmp) == 0:
        return None

    return tmp.iloc[0]['u']


def get_path(last_node, first_node, parents):
    res = [last_node]
    while last_node != first_node:
        last_node = parents[last_node]
        res.append(last_node)

    return list(reversed(res))


def find_shotest_path(start_node, end_node, graph: pd.DataFrame):
    start_id = get_node_id(*start_node, graph)
    end_id = get_node_id(*end_node, graph)

    start_end_distance = haversine_distance(*start_node, *end_node)

    g_score = dict()
    parents = dict()
    queue = PriorityQueue()

    # f_score, -g_score, node
    queue.put((start_end_distance, 0, start_id))
    g_score[start_id] = 0

    while not queue.empty():
        node_f_score, tmp, node = queue.get()
        node_g_score = -tmp
        if node_g_score > g_score[node]:
            continue

        if node == end_id:
            return get_path(end_id, start_id, parents)

        neighbours = get_neighbours(node, graph)
        for neighbour in neighbours:
            nei_h_score = neighbour.get_distance(*end_node)
            nei_g_score = node_g_score + neighbour.length
            nei_f_score = nei_g_score + nei_h_score
            nei_node = neighbour.key
            if nei_node in g_score and g_score[nei_node] < nei_g_score:
                continue
            g_score[nei_node] = nei_g_score
            parents[nei_node] = node
            queue.put((nei_f_score, -nei_g_score, nei_node))

    return None
