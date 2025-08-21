import os
import time
import urllib
from tfdp import tFDP  # NOQA: E402
import codingTree as ct
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import networkx as nx
import nodecolor
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection, PathCollection
import pandas as pd

import tarfile


def download_citeseer_to_local():
    base_url = "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"
    data_dir = "citeseer_data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    tgz_path = f"{data_dir}/citeseer.tgz"
    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(base_url, tgz_path)

    extract_dir = f"{data_dir}/citeseer"
    if not os.path.exists(extract_dir):
        tar = tarfile.open(tgz_path)
        tar.extractall(path=data_dir)
        tar.close()

    cites_path = f"{extract_dir}/citeseer.cites"
    content_path = f"{extract_dir}/citeseer.content"

    if not os.path.exists(cites_path) or not os.path.exists(content_path):
        raise FileNotFoundError(f"cannot find CiteSeer")

    return {"cites_path": cites_path, "content_path": content_path}


def load_citeseer_from_local(file_paths=None):
    if file_paths is None:
        data_dir = "citeseer_data"
        extract_dir = f"{data_dir}/citeseer"
        file_paths = {
            "cites_path": f"{extract_dir}/citeseer.cites",
            "content_path": f"{extract_dir}/citeseer.content"
        }

    if not os.path.exists(file_paths["cites_path"]) or not os.path.exists(file_paths["content_path"]):
        raise FileNotFoundError(f"Please download_citeseer_to_local()")

    cites = pd.read_csv(file_paths["cites_path"], sep='\t', header=None, names=['cited', 'citing'])

    content = pd.read_csv(file_paths["content_path"], sep='\t', header=None)

    num_columns = content.shape[1]

    G = nx.DiGraph()

    for i, row in content.iterrows():
        node_id = row[0]
        features = row[1:num_columns - 1].values
        label = row[num_columns - 1]
        G.add_node(node_id, features=features, label=label)

    for i, row in cites.iterrows():
        G.add_edge(row['cited'], row['citing'])
    return G

def download_cora_to_local():
    base_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    data_dir = "cora_data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    tgz_path = f"{data_dir}/cora.tgz"
    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(base_url, tgz_path)

    extract_dir = f"{data_dir}/cora"
    if not os.path.exists(extract_dir):
        tar = tarfile.open(tgz_path)
        tar.extractall(path=data_dir)
        tar.close()


    cites_path = f"{extract_dir}/cora.cites"
    content_path = f"{extract_dir}/cora.content"

    if not os.path.exists(cites_path) or not os.path.exists(content_path):
        raise FileNotFoundError(f"connot find Cora")

    print(f"cites_path: {cites_path}, content_path: {content_path}")
    return {"cites_path": cites_path, "content_path": content_path}


def load_cora_from_local(file_paths=None):
    if file_paths is None:
        data_dir = "cora_data"
        extract_dir = f"{data_dir}/cora"
        file_paths = {
            "cites_path": f"{extract_dir}/cora.cites",
            "content_path": f"{extract_dir}/cora.content"
        }

    if not os.path.exists(file_paths["cites_path"]) or not os.path.exists(file_paths["content_path"]):
        raise FileNotFoundError(f"Please download_cora_to_local()")

    cites = pd.read_csv(file_paths["cites_path"], sep='\t', header=None, names=['cited', 'citing'])

    content = pd.read_csv(file_paths["content_path"], sep='\t', header=None)

    num_columns = content.shape[1]

    G = nx.DiGraph()

    for i, row in content.iterrows():
        node_id = row[0]
        features = row[1:num_columns - 1].values
        label = row[num_columns - 1]
        G.add_node(node_id, features=features, label=label)

    for i, row in cites.iterrows():
        G.add_edge(row['cited'], row['citing'])

    return G

def json_to_networkx(json_data):
    """
    Convert a JSON structure to a NetworkX graph with node IDs mapped to 0, 1, 2, ...

    Parameters:
    - json_data: A dictionary that contains 'nodes', 'groups', and 'links'.

    Returns:
    - G: A NetworkX graph object with nodes and edges.
    """
    G = nx.Graph()  # Create an undirected graph (can be changed to nx.DiGraph for directed graph)

    # Create a mapping from the original node IDs to new numeric IDs (0, 1, 2, ...)
    id_mapping = {node['id']: idx for idx, node in enumerate(json_data['nodes'])}

    # Add nodes to the graph, using the numeric IDs instead of the original node IDs
    for idx, node in enumerate(json_data['nodes']):
        numeric_id = id_mapping[node['id']]  # Get the new numeric ID
        G.add_node(numeric_id, x=node['x'], y=node['y'], group=node['group'])

    # Add edges to the graph using the numeric IDs
    for link in json_data['links']:
        source = id_mapping[link['source']]  # Convert original source ID to numeric ID
        target = id_mapping[link['target']]  # Convert original target ID to numeric ID
        value = link['value']
        G.add_edge(source, target, weight=value)

    return G


def load_json_file(file_path):
    """
    Load a JSON file and return its content.

    Parameters:
    - file_path: The path to the JSON file.

    Returns:
    - data: The loaded JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
def tanh(x):
    return np.tanh(x)


def custom_logistic(x, k=10, b=0.8):
    if x >= 1:
        return 0
    elif x <= 0:
        return 1

    return 1 / (1 + np.exp(k * (x - b)))



def map_entropy_to_custom_logistic(attr_weight, x, k=10, b=0.8):
    mapped_values = {}

    for id, entropy in attr_weight.items():
        if x >= entropy:
            mapped_values[id] = float(0)
        else:
            value = 1 - x / entropy
            mapped_values[id] = float(custom_logistic(value, k, b))

    return mapped_values


def map_entropy_to_tanh(attr_weight, x):
    mapped_values = {}

    for id, entropy in attr_weight.items():
        if x >= entropy:
            mapped_values[id] = float(0)
        else:
            value = 1 - x / entropy
            mapped_values[id] = float((tanh(value) + 1) / 2)

    return mapped_values


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def map_entropy_to_sigmoid(attr_weight, x):
    mapped_values = {}

    for id, entropy in attr_weight.items():
        if x >= entropy:
            mapped_values[id] = float(0)
        else:
            value = 1 - x / entropy
            mapped_values[id] = float(sigmoid(value))

    return mapped_values

def map_entropy_to_01(attr_weight, x):
    mapped_values = {}

    for id, entropy in attr_weight.items():
        if x >= entropy:
            mapped_values[id] = float(0)
        else:
            mapped_values[id] = float(1)

    return mapped_values

def map_entropy_to_linear(attr_weight, x):
    mapped_values = {}
    for id, entropy in attr_weight.items():
        # 如果 x 大于 entropy，映射为 0
        if x >= entropy:
            mapped_values[id] = float(0)
        else:
            mapped_values[id] = float(1 - x / entropy)
    return mapped_values


def find_common_ancestor(node_dict, node_id1, node_id2):
    ancestors1 = set()
    current_node = node_dict[node_id1]
    while current_node is not None:
        ancestors1.add(current_node.ID)
        current_node = node_dict.get(current_node.parent)

    current_node = node_dict[node_id2]
    while current_node is not None:
        if current_node.ID in ancestors1:
            return current_node.ID
        current_node = node_dict.get(current_node.parent)

    return None


def extract_graphs_per_level(y, G_original):
    levels = traverse_tree_by_level(y)
    graphs = []
    for level_nodes in levels:
        G = nx.Graph()
        partition_mapping = {}
        for node_id, node in level_nodes.items():
            G.add_node(node_id, partition=node.partition, vol=node.vol, g=node.g)
            for u in node.partition:
                partition_mapping[u] = node_id
        edges = []
        for u, v in G_original.edges():
            u_partition = partition_mapping.get(u)
            v_partition = partition_mapping.get(v)
            if u_partition is not None and v_partition is not None and u_partition != v_partition:
                edges.append((u_partition, v_partition))
        edges = list(dict.fromkeys(edges))
        G.add_edges_from(edges)
        graphs.append(G)
    return graphs


def traverse_tree_by_level(y):
    if not y.tree_node:
        return []
    root_id = y.root_id
    if root_id not in y.tree_node:
        return []
    root_node = y.tree_node[root_id]
    levels = []
    current_level = {root_id: root_node}
    levels.append(current_level)
    queue = [root_node]
    while queue:
        next_level = {}
        next_queue = []
        for node in queue:
            for child_id in node.children:
                child = y.tree_node.get(child_id)
                if child is not None:
                    next_level[child_id] = child
                    next_queue.append(child)
        if next_level:
            levels.append(next_level)
            queue = next_queue
        else:
            queue = []
    return levels


def layout(G):
    H = relabel_graph_nodes_and_edges_with_attributes(G)
    pos_optimized = nx.spring_layout(H, seed=42)

    # using tfdp instead:
    # tfdp = tFDP(algo="ibFFT_CPU")
    # tfdp.init = "pmds"
    # tfdp.inputgraph(H)
    # tfdp.graphinfo()
    # pos_optimized, t = tfdp.optimization()

    return pos_optimized


def calculate_average_degree_by_label(G, label_attribute='group'):
    node_labels = [G.nodes[node].get(label_attribute) for node in G.nodes()]

    if None in node_labels:
        print("有些节点没有 '{}' 属性".format(label_attribute))
        return {}

    unique_labels = set(node_labels)

    average_degree = {}
    for label in unique_labels:
        nodes_with_label = [node for node in G.nodes() if G.nodes[node].get(label_attribute) == label]
        if not nodes_with_label:
            continue

        degrees = [G.degree[node] for node in nodes_with_label]

        avg = sum(degrees) / len(degrees)
        average_degree[label] = avg

    return average_degree

def weight_layout_partition_color(file, G_original, weight, entropy_list, func, layout=2):
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx

    H = relabel_graph_nodes_and_edges_with_attributes(G_original)

    # Preprocess node degrees
    node_degrees = dict(G_original.degree())
    min_degree = min(node_degrees.values())
    max_degree = max(node_degrees.values())

    # Initialize all nodes to the same cluster (0)
    clusters = {}
    for node in G_original.nodes():
        clusters[node] = 0

    # Track clusters created in each iteration
    all_clusters_by_iteration = [clusters.copy()]

    # Keep track of the next available cluster ID
    next_cluster_id = 1
    # Create a working copy of the graph
    G_working = G_original.copy()
    index = 0
    for ent in entropy_list:
        print(f"\nProcessing iteration {index} with entropy threshold: {ent}")

        # Map entropy values to weights using the selected function
        cur_weight = map_entropy_to_linear(weight, ent)
        if func == 1:
            cur_weight = map_entropy_to_sigmoid(weight, ent)
        if func == 2:
            cur_weight = map_entropy_to_tanh(weight, ent)
        if func == 3:
            cur_weight = map_entropy_to_custom_logistic(weight, ent)
        if func == 4:
            cur_weight = map_entropy_to_01(weight, ent)

        # Set the weights on the original graph
        nx.set_edge_attributes(G_original, cur_weight, 'weight')

        # Find all edges with weight EXACTLY equal to the current ent threshold
        cut_edges = []
        for u, v in G_original.edges():
            if (u, v) in weight:
                edge_weight = weight[(u, v)]
                # Use very small tolerance for floating point comparison
                if abs(edge_weight - ent) < 1e-10:
                    cut_edges.append((u, v))

        print(f"Number of cut edges at threshold {ent}: {len(cut_edges)}")



        # Remove all cut edges to identify connected components
        G_working.remove_edges_from(cut_edges)

        # Find connected components after removing cut edges
        components = list(nx.connected_components(G_working))
        print(f"Number of components after cutting: {len(components)}")

        # Group nodes by their current cluster ID
        cluster_nodes = {}
        for node, cluster_id in clusters.items():
            if cluster_id not in cluster_nodes:
                cluster_nodes[cluster_id] = set()
            cluster_nodes[cluster_id].add(node)

        # Create new clusters if components split existing clusters
        new_clusters = clusters.copy()

        # Process each existing cluster to see if it's split by the cut edges
        for cluster_id, nodes in cluster_nodes.items():
            # Find components that contain nodes from this cluster
            cluster_components = []
            for component in components:
                if component.intersection(nodes):
                    # Keep only the nodes that belong to this cluster
                    filtered_component = component.intersection(nodes)
                    cluster_components.append(filtered_component)

            # If the cluster is split into multiple components, assign new cluster IDs
            if len(cluster_components) > 1:
                print(f"Cluster {cluster_id} is split into {len(cluster_components)} parts")
                # Sort components by size (largest first)
                cluster_components.sort(key=len, reverse=True)

                # Largest component keeps the original cluster ID
                # Smaller components get new cluster IDs
                for i, component in enumerate(cluster_components):
                    if i > 0:  # Skip the largest component
                        for node in component:
                            new_clusters[node] = next_cluster_id
                        print(f"Created new cluster {next_cluster_id} with {len(component)} nodes")
                        next_cluster_id += 1
            else:
                print(f"Cluster {cluster_id} remains intact with {len(nodes)} nodes")

        # Update clusters for the next iteration
        clusters = new_clusters.copy()
        all_clusters_by_iteration.append(clusters.copy())

        print(f"Number of clusters after iteration {index}: {len(set(clusters.values()))}")

        # Create a color mapping for visualization
        unique_clusters = sorted(set(clusters.values()))
        num_clusters = len(unique_clusters)
        cmap = plt.cm.get_cmap('tab20', max(20, num_clusters))

        # Create a consistent color mapping
        cluster_colors = {cluster_id: cmap(i % cmap.N) for i, cluster_id in enumerate(unique_clusters)}

        # Choose layout algorithm
        if layout == 1:
            tfdp = tFDP(algo="ibFFT_CPU_aw")
            tfdp.init = "pmds"
            tfdp.weight = cur_weight
            tfdp.inputgraph(H)
            tfdp.graphinfo()
            pos_optimized, t = tfdp.optimization()
        if layout == 2:
            pos_optimized = nx.spring_layout(G_original, weight="weight")
        if layout == 3:
            tfdp = tFDP(algo="ibFFT_CPU")
            tfdp.init = "pmds"
            tfdp.inputgraph(H)
            tfdp.graphinfo()
            pos_optimized, t = tfdp.optimization()

        print("Rendering visualization...")

        # Create the figure
        plt.figure(figsize=(12, 12))

        # Draw edges with a consistent color
        for u, v in G_original.edges():
            plt.plot([pos_optimized[u][0], pos_optimized[v][0]],
                     [pos_optimized[u][1], pos_optimized[v][1]],
                     color='gray', linewidth=0.5, alpha=0.5, zorder=1)

        # Get node coordinates
        node_positions = np.array([pos_optimized[node] for node in G_original.nodes()])
        x, y = node_positions[:, 0], node_positions[:, 1]

        # Save coordinates
        print("Saving coordinates...")
        with open(f"../evaluation/{file}-{index}-coordinates.txt", 'w') as f:
            for i, node in enumerate(G_original.nodes()):
                f.write(f"{node} {x[i]} {y[i]}\n")

        # Node sizes based on degree
        node_sizes = []
        for node in G_original.nodes():
            degree = node_degrees[node]
            normalized_size = 50 + ((degree - min_degree) / (max_degree - min_degree + 0.1)) * 150
            node_sizes.append(normalized_size)

        # Node colors based on cluster
        node_colors = [cluster_colors[clusters[node]] for node in G_original.nodes()]

        # Draw nodes
        plt.scatter(x, y, s=node_sizes, c=node_colors, edgecolors='black', linewidths=1, zorder=2)

        # Save cluster information
        with open(f"../evaluation/{file}-{index}-clusters.txt", 'w') as f:
            for node in G_original.nodes():
                f.write(f"{node} {clusters[node]}\n")

        # Save the figure
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"../evaluation/{file}-{index}.png", bbox_inches='tight', dpi=300)
        plt.close()

        index += 1

    # Return all clustering results
    return all_clusters_by_iteration


def weight_layout(file,G_original, weight, entropy_list, func,layout=2):
    H = relabel_graph_nodes_and_edges_with_attributes(G_original)

    node_degrees = dict(G_original.degree())

    min_degree = min(node_degrees.values())
    max_degree = max(node_degrees.values())

    def degree_to_color(degree):
        normalized_degree = (degree - min_degree) / (max_degree - min_degree)
        color = mcolors.to_rgba((1, 1 - normalized_degree, 0))
        return color

    index=0
    for ent in entropy_list:
        cur_weight = map_entropy_to_linear(weight, ent)
        if func == 1:
            cur_weight = map_entropy_to_sigmoid(weight, ent)
        if func == 2:
            cur_weight = map_entropy_to_tanh(weight, ent)
        if func == 3:
            cur_weight = map_entropy_to_custom_logistic(weight, ent)
        if func == 4:
            cur_weight = map_entropy_to_01(weight, ent)
        nx.set_edge_attributes(G_original, cur_weight, 'weight')
        z_count=0
        for i in cur_weight:
            if cur_weight[i]==0:
                z_count+=1
                # print(i,cur_weight[i])
        print(f"z_count:{z_count}")
        if layout==1:
            tfdp = tFDP(algo="ibFFT_CPU_aw")
            tfdp.init = "pmds"
            tfdp.weight = cur_weight
            tfdp.inputgraph(H)
            tfdp.graphinfo()
            pos_optimized, t = tfdp.optimization()
        if layout == 2:
            pos_optimized = nx.spring_layout(G_original, weight="weight")
        if layout == 3:
            tfdp = tFDP(algo="ibFFT_CPU")
            tfdp.init = "pmds"
            tfdp.inputgraph(H)
            tfdp.graphinfo()
            pos_optimized, t = tfdp.optimization()
        # print(pos_optimized)
        # ds = dbscan_silhouette(pos_optimized)
        # tc = topology_consistency(pos_optimized, nx.to_numpy_array(G_original))
        # print(round(ds,3))
        print("rendering")
        edges = list(G_original.edges())

        edge_weights = np.array([G_original[u][v].get('weight', 1.0) for u, v in edges])
        min_weight, max_weight = edge_weights.min(), edge_weights.max()
        weight_range = max_weight - min_weight
        if weight_range == 0:
            normalized_weights = np.full_like(edge_weights, 0.6)  # 使用默认值
        else:
            normalized_weights = 0.2 + ((edge_weights - min_weight) / weight_range) * 0.8
        edge_colors = np.zeros((len(edges), 4))  # RGBA
        edge_colors[:, :3] = 0.5  # RGB
        edge_colors[:, 3] = normalized_weights  # Alpha

        plt.figure(figsize=(10, 10))

        edge_positions = np.array([(pos_optimized[u], pos_optimized[v]) for u, v in edges])
        line_collection = LineCollection(
            edge_positions,
            colors=edge_colors,
            linewidths=2,
            zorder=1
        )
        plt.gca().add_collection(line_collection)

        node_positions = np.array([pos_optimized[node] for node in G_original.nodes()])
        x, y = node_positions[:, 0], node_positions[:, 1]
        with open(f"../evaluation/{file}-{index}-coordinates.txt", 'w') as f:
            for i in range(len(x)):
                f.write(f"{x[i]} {y[i]}\n")
        node_sizes = 250


        if 'group' not in G_original.nodes[0] or check_node_group_attribute(G_original):
            node_colors = [degree_to_color(node_degrees[node]) for node in G_original.nodes()]
        else:
            print(f"group avg degree{calculate_average_degree_by_label(G_original)}")
            node_colors = [nodecolor.get_node_color(G_original,node) for node in G_original.nodes()]
        plt.scatter(x, y, s=node_sizes, c=node_colors, edgecolors='black', linewidths=1.5, zorder=2)
        plt.axis('off')
        plt.savefig(f"../evaluation/{file}-{index}.png", bbox_inches='tight')  # 使用 bbox_inches='tight' 避免裁剪

        index+=1
        plt.show()


def multilevel_layout(levels_graphs):
    pos_per_level = []  # Store layout for each level

    # Generate layout for each level
    for level in range(len(levels_graphs) - 1):
        H = relabel_graph_nodes_and_edges_with_attributes(levels_graphs[level])
        # pos_optimized = nx.spring_layout(H, seed=42)

        tfdp = tFDP(algo="ibFFT_CPU")
        tfdp.init = "pmds"
        tfdp.inputgraph(H)
        tfdp.graphinfo()
        pos_optimized, t = tfdp.optimization()

        pos_per_level.append(pos_optimized)

    return pos_per_level


def add_parent_children_attributes(levels_graphs):
    # Initialize parent and children attributes for all nodes
    for level in range(len(levels_graphs)):
        H = levels_graphs[level]
        for node in H.nodes():
            H.nodes[node]['parent'] = None
            H.nodes[node]['children'] = []

    # Traverse each level starting from the second level to determine parent-child relationships
    for current_level in range(1, len(levels_graphs)):
        current_H = levels_graphs[current_level]
        previous_H = levels_graphs[current_level - 1]

        for current_node in current_H.nodes():
            current_partition = set(current_H.nodes[current_node].get('partition', []))
            parent_node = None

            # Find the parent node in the previous level whose partition contains the current node's partition
            for prev_node in previous_H.nodes():
                prev_partition = set(previous_H.nodes[prev_node].get('partition', []))
                if current_partition.issubset(prev_partition):
                    parent_node = prev_node
                    break

            if parent_node is not None:
                # Assign the parent to the current node
                current_H.nodes[current_node]['parent'] = parent_node
                # Update the parent's children list in the previous level
                previous_H.nodes[parent_node]['children'].append(current_node)

    return levels_graphs


def plot_multilevel_layout(levels_graphs, pos_per_level, G):
    # Initialize a dictionary to store the parent node colors
    parent_color_map = {}

    # Traverse each level and assign colors to parent nodes
    for level in range(len(levels_graphs)):
        H = levels_graphs[level]

        for node in H.nodes():
            parent = H.nodes[node].get('parent', None)
            if parent is not None:
                if parent not in parent_color_map:
                    parent_color_map[parent] = np.random.rand(3, )  # Assign a random color to each parent

    # Create a color map for each node based on its parent
    for level in range(len(levels_graphs) - 1):
        H = relabel_graph_nodes_and_edges_with_attributes(levels_graphs[level])
        pos = pos_per_level[level]

        node_colors = []
        node_edge_colors = []
        node_edge_widths = []
        edge_colors = []  # To store the colors for edges
        transparent_edges = []  # Store transparent edges for later drawing

        # Traverse each edge in H and compare it with G
        for node in H.nodes():
            parent = H.nodes[node].get('parent', None)
            if parent is not None:
                # Node color is based on the current node
                node_colors.append(parent_color_map[H.nodes[node].get('cur', None)])  # Random color for the node itself
                # Edge color (node border) is based on the parent
                node_edge_colors.append(parent_color_map[parent])
                # Edge width is set to 4 for emphasis (thicker edge)
                node_edge_widths.append(10)
            else:
                # For root nodes, just set the color and edge color
                node_colors.append(parent_color_map[H.nodes[node].get('cur', None)])
                node_edge_colors.append(parent_color_map[H.nodes[node].get('cur', None)])
                node_edge_widths.append(2)

        # Now, for each edge in G that is not in H, make the edge very transparent
        for u, v in G.edges():
            if not H.has_edge(u, v):
                # Store transparent edges
                transparent_edges.append((pos[u], pos[v]))
            else:
                # Regular edge color for edges that are in both H and G
                edge_colors.append('gray')  # Set the color to gray for these edges

        # Draw the graph
        plt.figure(figsize=(12, 12))

        # First, draw all the edges
        nx.draw_networkx_edges(
            H, pos,
            edge_color=edge_colors,  # Set edge colors for regular edges
            width=1,
            alpha=1.0
        )

        # Then, draw the transparent edges (edges that are in G but not in H)
        if transparent_edges:
            lc = LineCollection(transparent_edges, linewidths=1, colors='gray', alpha=0.1)
            plt.gca().add_collection(lc)

        # Now, draw the nodes on top of the edges
        nx.draw_networkx_nodes(
            H, pos,
            node_color=node_colors,
            node_size=200,
            edgecolors=node_edge_colors,  # Set node border color
            linewidths=4  # Make the node border thicker
        )

        plt.title(f"Level {level} Layout with Parent and Node Border Colors")
        plt.show()


def relabel_graph_nodes_and_edges_with_attributes(G):
    original_node_attributes = {node: G.nodes[node] for node in G.nodes()}

    nodes = list(G.nodes())

    node_mapping = {node: i for i, node in enumerate(nodes)}

    new_G = nx.Graph()

    new_G.add_nodes_from(range(len(nodes)))
    for old_node, new_node in node_mapping.items():
        new_G.nodes[new_node].update(original_node_attributes[old_node])

    new_edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]
    new_G.add_edges_from(new_edges)

    return new_G


import heapq
from collections import deque

class PriorityNode_acc:
    def __init__(self, node_id, accumulated_entropy):
        self.node_id = node_id
        self.accumulated_entropy = accumulated_entropy

    def __lt__(self, other):
        return self.accumulated_entropy < other.accumulated_entropy


class PriorityNode:
    def __init__(self, node_id, entropy, level):
        self.node_id = node_id
        self.entropy = entropy
        self.level = level

    def __lt__(self, other):
        if self.level == other.level:
            return self.entropy < other.entropy
        return self.level < other.level


def check_child(node,y):
    j=True
    if node.children:
        for child_id in node.children:
            if len(y.tree_node[child_id].partition) == 1:
                j = False
                break
    else:
        j=False
    return j
def attr_weight_with_entropy(y, G):
    priority_queue = []
    attr_weight = {}
    for u, v in G.edges():
        attr_weight[(u, v)] = 100
    root_id = y.root_id
    node_dict = {root_id: y.tree_node[root_id]}
    entropy_list=[]
    heapq.heappush(priority_queue, PriorityNode_acc(root_id, 0))
    queue_num=0
    while priority_queue:
        current_node = heapq.heappop(priority_queue)

        node_id = current_node.node_id
        node = y.tree_node[node_id]
        print(node_id,current_node.accumulated_entropy,node.children)
        if len(y.tree_node[node_id].partition) > 1:
            node.cur_E = y.entropy(node_dict)
            child_dict={}
            for children_id in node.children:
                for p in y.tree_node[children_id].partition:
                    child_dict[p]=children_id
            inter_edge_num=0
            for u,v in G.edges():
                if u in child_dict and v in child_dict:
                    if child_dict[u] != child_dict[v]:
                        inter_edge_num+=1
                        attr_weight[(u, v)] = float(node.cur_E)

            if not check_child(node,y):
                heapq.heappush(priority_queue, PriorityNode_acc(current_node.node_id, 100))
            elif node.children:
                for child_id in node.children:
                    node_dict[child_id] = y.tree_node[child_id]
                    if y.tree_node[child_id].children:
                        heapq.heappush(priority_queue, PriorityNode_acc(child_id, y.tree_node[child_id].A_increase))
                if node.cur_E not in entropy_list:
                    entropy_list.append(node.cur_E)


        node_partition = {}
        for acc_node in priority_queue:
            for node_id in y.tree_node[acc_node.node_id].partition:
                node_partition[node_id] = acc_node.node_id



        queue_num += 1
        if queue_num >= 10 or out_queue(priority_queue):
            print(queue_num)
            break
    return attr_weight, entropy_list

def out_queue(queue):
    quit=True
    for i in range(len(queue)):
        if queue[i].accumulated_entropy!=100:
            quit=False
    return quit
def attr_weight_with_level_entropy(y, G, level):
    attr_weight = {}
    root_id = y.root_id
    node_dict = {root_id: y.tree_node[root_id]}
    entropy_list=[]

    for l in range(level):
        level_node_dict = {}
        for id in y.tree_node:
            if y.tree_node[id].depth == l:
                node_dict[id] = y.tree_node[id]
                level_node_dict[id] = y.tree_node[id]

        current_entropy = y.entropy(node_dict)
        if current_entropy not in entropy_list:
            entropy_list.append(current_entropy)

        for node_id in level_node_dict:
            for i in range(len(y.tree_node[node_id].partition)):
                for j in range(len(y.tree_node[node_id].partition)):
                    if y.tree_node[node_id].partition[i] <= y.tree_node[node_id].partition[j]:
                        if G.has_edge(y.tree_node[node_id].partition[i], y.tree_node[node_id].partition[j]):
                            attr_weight[
                                (
                                y.tree_node[node_id].partition[i], y.tree_node[node_id].partition[j])] = float(current_entropy)

        count_incluster = 0
        count_outcluster = 1
        node_partition = {}
        for k in level_node_dict.keys():
            for node in level_node_dict[k].partition:
                node_partition[node] = k

        for i, j in G.edges():
            if i == j: continue
            if node_partition.get(i) != node_partition.get(j):
                count_outcluster += 1
            else:
                count_incluster += 1
        in_out = count_incluster / count_outcluster
        print(f"level{l}:{in_out}")
        print(f"cluster:{len(level_node_dict)} ent:{current_entropy} score:{in_out}")
    return attr_weight, entropy_list

def relabel_nodes_to_0_to_N_minus_1(G):
    nodes = list(G.nodes())

    mapping = {nodes[i]: i for i in range(len(nodes))}

    G_relabelled = nx.relabel_nodes(G, mapping)

    return G_relabelled


def check_node_group_attribute(G, attribute_name='group'):
    node_groups = [G.nodes[node].get(attribute_name) for node in G.nodes()]

    if None in node_groups:
        return False

    unique_groups = set(node_groups)
    if len(unique_groups) == 1:
        return True
    else:
        return False



def read_attr_file(attr_file):
    with open(attr_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    return labels

def read_mtx_file(mtx_file):
    edges = []
    with open(mtx_file, 'r') as f:
        for line in f.readlines():
            u, v = map(int, line.strip().split())
            edges.append((u, v))
    return edges

def build_graph_attr(attr_file, mtx_file):
    node_labels = read_attr_file(attr_file)
    edges = read_mtx_file(mtx_file)

    G = nx.Graph()

    for node, label in enumerate(node_labels):
        G.add_node(node, group=label)

    G.add_edges_from(edges)

    return G

if __name__ == "__main__":
    G_class = -1
    G = nx.karate_club_graph()
    if G_class == -1:

        adj_sparse = nx.adjacency_matrix(G)
        adj_matrix = adj_sparse.toarray().astype(np.float32)
        y = ct.PartitionTree(adj_matrix)
        depth = 5
        y.build_coding_tree(k=depth, mode='v2')
        entropy_increases = y.calculate_entropy_increase()
        y.assign_colors_to_non_leaf_nodes(max_depth=depth)
        # 打印熵的增加量
        for node_id, increase in entropy_increases.items():
            y.tree_node[node_id].E_increase = increase
        acc_increases = y.calculate_accumulated_entropy()
        for node_id, increase in acc_increases.items():
            y.tree_node[node_id].A_increase = increase
            print(
                f"Node ID: {node_id}, height: {y.tree_node[node_id].depth}, Entropy Increase: {y.tree_node[node_id].E_increase}, ACC Increase: {y.tree_node[node_id].A_increase},color: {y.node_color[node_id]}，partition: {len(y.tree_node[node_id].partition)}")
        func = 1  # 0: linear; 1: sigmoid; 2:tanh; 3: logistic 4:01
        version = 0  # queue or level
        print("start layout")
        if version == 0:
            attr_weight, entropy_list = attr_weight_with_entropy(y, G)
            # print(attr_weight)
            # print(entropy_list)
            weight_layout("club", G, attr_weight, entropy_list, func)
        if version == 1:
            attr_weight, entropy_list = attr_weight_with_level_entropy(y, G, depth)
            # print(attr_weight)
            print(entropy_list)
            weight_layout("club-level", G, attr_weight, entropy_list, func)
    if G_class == 0:
        # filename = ["socfb-Simmons81", "socfb-Reed98","socfb-Haverford76","socfb-Swarthmore42","socfb-USFCA72","socfb-Smith60","socfb-Trinity100","socfb-Rice31", "socfb-American75", "socfb-WashU32"]
        filename = ["karate_club"]
        for i in range(len(filename)):
            # G_data = nx.read_edgelist(f'../facebook100/{filename[i]}.txt', create_using=nx.Graph(), nodetype=int)
            # largest_component = max(nx.connected_components(G_data), key=len)
            # G_sub = G_data.subgraph(largest_component).copy()
            # G = relabel_nodes_to_0_to_N_minus_1(G_sub)

            pos = nx.spring_layout(G)

            edges = list(G.edges())

            edge_weights = np.array([G[u][v].get('weight', 1.0) for u, v in edges])
            min_weight, max_weight = edge_weights.min(), edge_weights.max()
            weight_range = max_weight - min_weight
            if weight_range == 0:
                normalized_weights = np.full_like(edge_weights, 0.6)
            else:
                normalized_weights = 0.2 + ((edge_weights - min_weight) / weight_range) * 0.8
            edge_colors = np.zeros((len(edges), 4))  # RGBA
            edge_colors[:, :3] = 0.5  # 灰色RGB
            edge_colors[:, 3] = normalized_weights  # Alpha

            plt.figure(figsize=(10, 10))


            edge_positions = np.array([(pos[u], pos[v]) for u, v in edges])
            line_collection = LineCollection(
                edge_positions,
                colors=edge_colors,
                linewidths=2,
                zorder=1
            )
            plt.gca().add_collection(line_collection)

            node_positions = np.array([pos[node] for node in G.nodes()])
            x, y = node_positions[:, 0], node_positions[:, 1]
            node_sizes = 300

            node_degrees = dict(G.degree())
            min_degree = min(node_degrees.values())
            max_degree = max(node_degrees.values())

            def degree_to_color(degree):
                normalized_degree = (degree - min_degree) / (max_degree - min_degree)
                color = mcolors.to_rgba((1, 1 - normalized_degree, 0))  # 黄色到红色
                return color
            node_colors = [degree_to_color(node_degrees[node]) for node in G.nodes()]
            plt.scatter(x, y, s=node_sizes, c=node_colors, edgecolors='black', linewidths=1.5, zorder=2)
            # for i, node in enumerate(G_original.nodes()):
            #     plt.text(x[i], y[i], str(node), fontsize=8, ha='center', va='center', color='black', zorder=3)
            plt.axis('off')
            plt.show()
    if G_class == 1:
        # filename = "socfb-Simmons81.txt,"
        # filename = ["socfb-Simmons81", "socfb-Reed98","socfb-Haverford76","socfb-Swarthmore42","socfb-USFCA72","socfb-Smith60","socfb-Trinity100","socfb-Rice31", "socfb-American75", "socfb-WashU32"]
        filename = ["facebook_4039"]
        for i in range(len(filename)):
            G_data = nx.read_edgelist(f'../facebook100/{filename[i]}.txt', create_using=nx.Graph(), nodetype=int)
            largest_component = max(nx.connected_components(G_data), key=len)
            G_sub = G_data.subgraph(largest_component).copy()
            G = relabel_nodes_to_0_to_N_minus_1(G_sub)
            print("start coding_tree")
            start_time = time.time()
            adj_sparse = nx.adjacency_matrix(G)
            adj_matrix = adj_sparse.toarray().astype(np.float32)
            y = ct.PartitionTree(adj_matrix)
            depth = 5
            y.build_coding_tree(k=depth, mode='v2')
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"{elapsed_time} s")
            func = 1  # 0: linear; 1: sigmoid; 2:tanh; 3: logistic 4:01
            version = 0  # queue or level
            print("start layout")
            if version == 0:
                attr_weight, entropy_list = attr_weight_with_entropy(y, G)
                # print(attr_weight)
                # print(entropy_list)
                weight_layout_partition_color(filename[i],G, attr_weight, entropy_list, func)
            if version == 1:
                attr_weight, entropy_list = attr_weight_with_level_entropy(y, G, depth)
                # print(attr_weight)
                print(entropy_list)
                weight_layout_partition_color(filename[i],G, attr_weight, entropy_list, func)
    if G_class == 2:
        # filename = ["les", "airport", "caltech", "smith", "collab.science","map.science"]
        filename = ["les"]
        for i in range(len(filename)):
            # try:
                test_data=filename[0]
                file_path = f'../datasets/{test_data}.json'  # Specify the correct file path
                json_data = load_json_file(file_path)
                G_data = json_to_networkx(json_data)
                largest_component = max(nx.connected_components(G_data), key=len)
                G_sub = G_data.subgraph(largest_component).copy()
                G = relabel_nodes_to_0_to_N_minus_1(G_sub)
                # G = json_to_networkx(json_data)
                print("start coding_tree")
                start_time = time.time()
                adj_sparse = nx.adjacency_matrix(G)
                adj_matrix = adj_sparse.toarray().astype(np.float32)
                y = ct.PartitionTree(adj_matrix)
                depth = 5
                # 构建分区树
                y.build_coding_tree(k=depth, mode='v2')
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"{elapsed_time} s")
                entropy_increases = y.calculate_entropy_increase()
                y.assign_colors_to_non_leaf_nodes(max_depth=depth)
                for node_id, increase in entropy_increases.items():
                    y.tree_node[node_id].E_increase = increase
                acc_increases = y.calculate_accumulated_entropy()
                for node_id, increase in acc_increases.items():
                    y.tree_node[node_id].A_increase = increase
                    print(
                        f"Node ID: {node_id}, height: {y.tree_node[node_id].depth}, Entropy Increase: {y.tree_node[node_id].E_increase}, ACC Increase: {y.tree_node[node_id].A_increase},children: {y.tree_node[node_id].children}，partition: {len(y.tree_node[node_id].partition)}")
                end_time = time.time()

                # 计算耗时
                elapsed_time = end_time - start_time
                print(f"{elapsed_time} s")
                func = 1  # 0: linear; 1: sigmoid; 2:tanh; 3: logistic 4:01
                version = 0  # queue or level
                print("start layout")
                if version == 0:
                    attr_weight, entropy_list = attr_weight_with_entropy(y, G)
                    # print(attr_weight)
                    # print(entropy_list)
                    weight_layout(test_data,G, attr_weight, entropy_list, func)
                if version == 1:
                    attr_weight, entropy_list = attr_weight_with_level_entropy(y, G, depth)
                    # print(attr_weight)
                    # print(entropy_list)
                    weight_layout(test_data,G, attr_weight, entropy_list, func)

    if G_class == 3:
        filename=["co_author_8391", "ACO","APH"]
        for i in range(len(filename)):
            test_data = filename[i]
            attr_file = f"../labeled_data/{test_data}.attr"
            mtx_file = f"../labeled_data/{test_data}.mtx"

            G = build_graph_attr(attr_file, mtx_file)
            print("start coding_tree")
            start_time = time.time()
            adj_sparse = nx.adjacency_matrix(G)
            adj_matrix = adj_sparse.toarray().astype(np.float32)
            y = ct.PartitionTree(adj_matrix)
            depth = 5
            y.build_coding_tree(k=depth, mode='v2')
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"{elapsed_time} s")
            entropy_increases = y.calculate_entropy_increase()
            y.assign_colors_to_non_leaf_nodes(max_depth=depth)
            for node_id, increase in entropy_increases.items():
                y.tree_node[node_id].E_increase = increase
            acc_increases = y.calculate_accumulated_entropy()
            for node_id, increase in acc_increases.items():
                y.tree_node[node_id].A_increase = increase
                print(
                    f"Node ID: {node_id}, height: {y.tree_node[node_id].depth}, Entropy Increase: {y.tree_node[node_id].E_increase}, ACC Increase: {y.tree_node[node_id].A_increase},color: {y.node_color[node_id]}，partition: {len(y.tree_node[node_id].partition)}")
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"{elapsed_time} s")
            func = 1  # 0: linear; 1: sigmoid; 2:tanh; 3: logistic 4:01
            version = 0  # queue or level
            print("start layout")
            if version == 0:
                attr_weight, entropy_list = attr_weight_with_entropy(y, G)
                # print(attr_weight)
                # print(entropy_list)
                weight_layout(test_data, G, attr_weight, entropy_list, func)
            if version == 1:
                attr_weight, entropy_list = attr_weight_with_level_entropy(y, G, depth)
                # print(attr_weight)
                # print(entropy_list)
                weight_layout(test_data, G, attr_weight, entropy_list, func)