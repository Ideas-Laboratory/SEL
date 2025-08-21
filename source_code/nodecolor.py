import matplotlib.pyplot as plt
import networkx as nx
class QualitativeColormap:
    def __init__(self, *colors):
        self.colors = colors

    def get_color(self, index):
        return self.colors[index % len(self.colors)]

default_color_map = QualitativeColormap(
    0xffa6cee3, 0xff1f78b4, 0xffb2df8a, 0xff33a02c,
    0xfffb9a99, 0xffe31a1c, 0xff8C564B, 0xffff7f00,
    0xffcab2d6, 0xff6a3d9a, 0xffffff99, 0xffb15928
)

science_color_map = QualitativeColormap(
    0xff99ccff, 0xff1f78b4, 0xff97bc4e, 0xff007b0c,
    0xffff9b98, 0xffcc0000, 0xfff5a33e, 0xffff7619,
    0xffbfabd2, 0xff562f7e, 0xffffeb7a, 0xff8e5119,
    0xff670a0a
)

senate_color_map = QualitativeColormap(
    0xffa6cee3, 0xff1f78b4, 0xffb2df8a, 0xff33a02c,
    0xfffb9a99, 0xffe31a1c, 0xff8C564B, 0xffff7f00,
    0xffcab2d6, 0xff6a3d9a, 0xffffff99, 0xffb15928
)

cur_color_map = default_color_map

def switch_color_map(color_id):
    global cur_color_map
    if color_id == 0:
        cur_color_map = default_color_map
    elif color_id == 1:
        cur_color_map = science_color_map
    elif color_id == 2:
        cur_color_map = senate_color_map

def get_node_color(G,node, color_by_degree=False, min_degree=0, max_degree=100):
    if color_by_degree:
        degree = G.degree[node]
        # 根据度数计算颜色
        red = int(map_value(degree, min_degree, max_degree, 255, 240))
        green = int(map_value(degree, min_degree, max_degree, 255, 59))
        blue = int(map_value(degree, min_degree, max_degree, 178, 32))
        return (red / 255.0, green / 255.0, blue / 255.0)
    else:
        group = G.nodes[node].get('group', 0)
        color = cur_color_map.get_color(group)
        return (color >> 16 & 0xFF) / 255.0, (color >> 8 & 0xFF) / 255.0, (color & 0xFF) / 255.0

def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
