import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from itertools import combinations

def reposition_nodes_to_circle(subgraph, center, radius):
    """
    将子图的节点重新分布到圆周上：
    1. 节点沿中心到其位置的方向移动到圆周上。
    2. 按顺时针顺序重新排列节点。
    """
    # 提取子图的所有节点
    nodes = list(subgraph.nodes())
    if len(nodes) == 0:
        return {}

    # 获取原始节点坐标
    original_positions = nx.get_node_attributes(subgraph, 'pos')

    # Step 1: 将节点移动到圆周上
    moved_positions = {}
    for node in nodes:
        x, y = original_positions[node]  # 原始位置
        dx, dy = x - center[0], y - center[1]  # 到中心的向量
        norm = np.sqrt(dx ** 2 + dy ** 2)  # 向量长度
        if norm == 0:
            # 避免节点恰好在中心点的情况
            dx, dy = 1.0, 0.0
            norm = 1.0
        # 计算新的位置
        new_x = center[0] + radius * (dx / norm)
        new_y = center[1] + radius * (dy / norm)
        moved_positions[node] = (new_x, new_y)

    # Step 2: 按顺时针顺序重新排列节点
    def compute_angle(pos):
        """计算点相对于圆心的极角（0到2π范围内）。"""
        x, y = pos[0] - center[0], pos[1] - center[1]
        angle = np.arctan2(y, x)
        return angle if angle >= 0 else (2 * np.pi + angle)

    # 计算每个节点的角度
    angles = {node: compute_angle(moved_positions[node]) for node in nodes}
    # 按角度排序节点
    sorted_nodes = sorted(nodes, key=lambda n: angles[n])

    # Step 3: 按顺时针顺序重新分布节点
    final_positions = {}
    angle_step = 2 * np.pi / len(sorted_nodes)  # 平均角度间隔
    for i, node in enumerate(sorted_nodes):
        angle = i * angle_step
        new_x = center[0] + radius * np.cos(angle)
        new_y = center[1] + radius * np.sin(angle)
        final_positions[node] = (new_x, new_y)

    return final_positions


def assign_pack_layout_v3(G, supernode_to_subgraph, final_pos, node_labels, center_list, range_list):
    """
    改进版布局算法：
    1. 仅根据外部连接优化角度
    2. 保持半径层级约束
    3. 最小化跨子图边长度
    """
    node_positions = {}
    radius_scale = {}

    # ====== 节点角度优化 ======
    for sg_idx, (nodes, center, max_r) in enumerate(zip(
            supernode_to_subgraph.values(),
            center_list,
            range_list
    )):
        # 提取当前子图节点集合
        current_subgraph_nodes = set(nodes)

        # 生成层级半径映射（保持不变）
        labels = [node_labels[n] for n in nodes]
        min_label = min(labels, default=0)
        max_label = max(labels, default=0)
        layer_radii = {}
        if min_label == max_label:
            layer_radii = {min_label: max_r}
        else:
            for label in set(labels):
                ratio = (label - min_label) / (max_label - min_label)
                layer_radii[label] = max_r * (1 - 0.5 * ratio)

        # 按标签分层处理
        layers = {}
        for n in nodes:
            lbl = node_labels[n]
            layers.setdefault(lbl, []).append(n)

        # 优化每层节点的角度分布
        for label, members in layers.items():
            r = layer_radii[label]

            # 步骤1：计算理想角度（仅考虑外部边）
            ideal_angles = []
            for node in members:
                vec_sum = np.zeros(2)

                # 仅处理连接到其他子图的边
                for neighbor in G.neighbors(node):
                    if neighbor not in current_subgraph_nodes:
                        nx, ny = final_pos[neighbor]
                        dx = nx - center[0]
                        dy = ny - center[1]
                        vec_sum += np.array([dx, dy])

                # 计算最优角度（指向外部连接的反方向）
                if np.linalg.norm(vec_sum) > 1e-6:
                    theta = np.arctan2(vec_sum[1], vec_sum[0]) + np.pi
                else:
                    # 如果没有外部连接，使用随机角度
                    theta = np.random.uniform(0, 2 * np.pi)

                ideal_angles.append(theta)

            # 步骤2：角度排序并均匀化（保持不变）
            sorted_indices = np.argsort(ideal_angles)
            sorted_nodes = [members[i] for i in sorted_indices]
            base_angles = np.linspace(0, 2 * np.pi, len(members) + 1)[:-1]
            angle_mapping = {n: base_angles[i] for i, n in enumerate(sorted_nodes)}

            # 步骤3：最终角度分配（调整混合权重）
            for node in members:
                ideal = ideal_angles[members.index(node)]
                uniform = angle_mapping[node]

                # 根据外部连接数量动态调整权重
                external_degree = len([n for n in G.neighbors(node) if n not in current_subgraph_nodes])
                alpha = 0.3 + 0.7 * (external_degree / (G.degree(node) + 1e-6))  # 外部连接越多越侧重优化
                alpha = 1
                final_angle = alpha * ideal + (1 - alpha) * uniform
                x = center[0] + r * np.cos(final_angle)
                y = center[1] + r * np.sin(final_angle)

                node_positions[node] = (x, y)
                radius_scale[node] = r
                G.nodes[node]['pos'] = (x, y)
                final_pos[node] = (x, y)

    return node_positions, radius_scale


def assign_pack_layout_v2(G, supernode_to_subgraph, final_pos, node_labels, center_list, range_list):
    """
    改进版布局算法：
    1. 根据节点原始位置动态计算子图中心
    2. 自动调整子图半径避免重叠
    3. 层级半径反向分配（小标签大半径）
    """
    node_positions = {}
    radius_scale = {}

    # ====== 更新center_list和range_list ======
    # 步骤1：重新计算子图中心点和初始半径
    for sg_idx, nodes in enumerate(supernode_to_subgraph.values()):
        # 计算几何中心
        coords = np.array([final_pos[node] for node in nodes])
        new_center = coords.mean(axis=0)
        center_list[sg_idx] = tuple(new_center)

        # 计算最大半径
        offsets = coords - new_center
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        range_list[sg_idx] = distances.max() if len(distances) > 0 else 0

    # 步骤2：调整子图半径避免重叠
    sg_indices = list(range(len(center_list)))
    for i, j in combinations(sg_indices, 2):
        # 计算子图间距
        dx = center_list[j][0] - center_list[i][0]
        dy = center_list[j][1] - center_list[i][1]
        distance = np.hypot(dx, dy)

        # 计算半径和
        r_sum = range_list[i] + range_list[j]

        if r_sum > distance:
            # 等比例压缩半径
            scale = distance / r_sum * 0.9  # 保留10%间距
            range_list[i] *= scale
            range_list[j] *= scale

    # ====== 生成节点位置 ======
    for sg_idx, (nodes, center, max_r) in enumerate(zip(
            supernode_to_subgraph.values(),
            center_list,
            range_list
    )):
        # 提取标签范围
        labels = [node_labels[n] for n in nodes]
        min_label = min(labels, default=0)
        max_label = max(labels, default=0)

        # 生成层级半径映射
        layer_radii = {}
        if min_label == max_label:
            layer_radii = {min_label: max_r}
        else:
            for label in set(labels):
                ratio = (label - min_label) / (max_label - min_label)
                layer_radii[label] = max_r * (1 - 0.5 * ratio)  # 从max_r到0.5max_r

        # 按标签分层布局
        layers = {}
        for n in nodes:
            lbl = node_labels[n]
            layers.setdefault(lbl, []).append(n)

        # 严格同心圆分布
        for label, members in layers.items():
            r = layer_radii[label]
            angles = np.linspace(0, 2 * np.pi, len(members) + 1)[:-1]

            for i, node in enumerate(members):
                x = center[0] + r * np.cos(angles[i])
                y = center[1] + r * np.sin(angles[i])

                node_positions[node] = (x, y)
                radius_scale[node] = r
                G.nodes[node]['pos'] = (x, y)
                final_pos[node] = (x, y)

    return node_positions, radius_scale
def assign_pack_layout(G, supernode_to_subgraph, final_pos, center_dict, node_labels, range_list):
    """
    改进版布局算法：
    - 基于原始final_pos的位置
    - 每个子图的节点布局受range_list中的半径范围限制
    - 确保节点不会超出子图的半径范围
    """
    node_positions = {}
    radius_scale = {}

    for subgraph_id, nodes in supernode_to_subgraph.items():
        # 获取子图的中心坐标和半径范围
        center_x, center_y = center_dict[subgraph_id]
        max_radius = range_list[subgraph_id]  # 从range_list中获取当前子图的最大半径

        # 按节点标签分层
        layers = {}
        for node in nodes:
            label = node_labels[node]
            if label not in layers:
                layers[label] = []
            layers[label].append(node)

        # 计算每层的半径
        max_label = max(layers.keys(), default=0)
        radius_step = max_radius / (max_label + 1)  # 每层的半径增量

        # 为每层分配位置
        for label, layer_nodes in layers.items():
            # 计算当前层的半径
            layer_radius = radius_step * (label + 1)

            # 均匀分布节点
            angle_step = 2 * np.pi / len(layer_nodes)
            for i, node in enumerate(layer_nodes):
                # 原始位置
                orig_x, orig_y = final_pos[node]

                # 计算目标位置（围绕中心均匀分布）
                target_x = center_x + layer_radius * np.cos(i * angle_step)
                target_y = center_y + layer_radius * np.sin(i * angle_step)

                # 计算偏移量
                dx = target_x - orig_x
                dy = target_y - orig_y
                shift_distance = np.sqrt(dx**2 + dy**2)

                # 限制偏移量不超过max_radius
                if shift_distance > max_radius:
                    scale_factor = max_radius / shift_distance
                    dx *= scale_factor
                    dy *= scale_factor

                # 计算最终位置
                new_x = orig_x + dx
                new_y = orig_y + dy

                # 更新节点位置
                node_positions[node] = (new_x, new_y)
                radius_scale[node] = layer_radius * 0.5  # 可视化的半径大小

                # 更新图数据和最终位置
                G.nodes[node]['pos'] = (new_x, new_y)
                final_pos[node] = (new_x, new_y)

    return node_positions, radius_scale


def alignment(G, supernode_to_subgraph, final_pos, center_list, range_list):
    # 遍历每个超点的子图，将 final_pos 中的坐标填入节点
    supernode_i = 0
    for supernode, subgraph in supernode_to_subgraph.items():
        for node in subgraph.nodes():
            # 将 final_pos 中的坐标先更新到子图中
            subgraph.nodes[node]['pos'] = final_pos[node]

        # 获取当前子图的圆心和半径
        center = center_list[supernode_i]  # 圆心
        radius = range_list[supernode_i]  # 半径
        supernode_i += 1

        # 调用重新分布函数
        sub_pos_dict = reposition_nodes_to_circle(subgraph, center, radius)

        # 同步子图中节点的位置到 G 中
        for node, pos in sub_pos_dict.items():
            G.nodes[node]['pos'] = pos
            final_pos[node] = pos
        # pos_dict = {node: [0.0, 0.0] for node in G.nodes()}
        # for i in range(len(G.nodes())):
        #     pos_dict[i] = G.nodes[i]['pos']
        # node_colors = ['red' if node in node_to_supernode else 'lightblue' for node in G.nodes()]
        # nx.draw(G, pos_dict, with_labels=True, node_color=node_colors, edge_color='gray')
        # plt.title("Multilevel Layout with super nodes")
        # plt.show()
        # plt.close()


def iterative_assign_pack_layout(G, supernode_to_subgraph, final_pos, node_labels, max_iter=5):
    """
    迭代优化版布局算法：
    1. 动态计算中心点和半径
    2. 多轮迭代优化布局
    3. 自动收敛检测
    """
    # 初始化中心点和半径列表
    center_list = []
    range_list = []

    for _ in range(max_iter):
        # ====== 更新中心点和半径 ======
        center_list.clear()
        range_list.clear()

        # 计算当前迭代的中心点和半径
        for nodes in supernode_to_subgraph.values():
            coords = np.array([final_pos[node] for node in nodes])
            center = coords.mean(axis=0)
            offsets = coords - center
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
            max_r = distances.max() if len(distances) > 0 else 0

            center_list.append(tuple(center))
            range_list.append(max_r)

        # ====== 调整子图间距 ======
        for i, j in combinations(range(len(center_list)), 2):
            dx = center_list[j][0] - center_list[i][0]
            dy = center_list[j][1] - center_list[i][1]
            distance = np.hypot(dx, dy)
            r_sum = range_list[i] + range_list[j]

            if r_sum > distance * 0.9:
                scale = (distance * 0.9) / r_sum
                range_list[i] *= scale
                range_list[j] *= scale

        # ====== 优化节点角度 ======
        assign_pack_layout(G, supernode_to_subgraph, final_pos, node_labels, center_list, range_list)

        # ====== 收敛检测 ======
        if check_convergence(final_pos):  # 需要实现收敛检测函数
            break

    return final_pos


def check_convergence(positions, threshold=0.1):
    """
    简易收敛检测：连续两次位置变化小于阈值
    """
    global prev_positions  # 需要维护全局状态记录前次位置
    if 'prev_positions' not in globals():
        prev_positions = positions.copy()
        return False

    max_change = max(
        np.hypot(x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in zip(prev_positions.values(), positions.values())
    )
    prev_positions = positions.copy()
    return max_change < threshold