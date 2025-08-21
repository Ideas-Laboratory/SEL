import json
import os
import tarfile
import urllib
import pandas as pd
import numpy as np
import networkx as nx
import copy
import heapq
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import math
def get_id():
    i = 0
    while True:
        yield i
        i += 1
def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = {}
    VOL = 0
    node_vol = []
    for i in range(g_num_nodes):
        n_v = 0
        adj = set()
        for j in range(g_num_nodes):
            if adj_matrix[i,j] != 0:
                n_v += adj_matrix[i,j]
                VOL += adj_matrix[i,j]
                # n_v += 1
                # VOL += 1
                adj.add(j)
        adj_table[i] = adj
        node_vol.append(n_v)
    return g_num_nodes,VOL,node_vol,adj_table

@nb.jit(nopython=True)
def cut_volume(adj_matrix,p1,p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            c = adj_matrix[p1[i],p2[j]]
            if c != 0:
                c12 += c
    return c12

def LayerFirst(node_dict,start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)


def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h,node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID,partition=new_partition,children={id1,id2},
                                 g=g, vol=v,child_h= child_h,child_cut = cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node


def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([node_dict[f_c].child_h for f_c in node_dict[parent_id].children])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break



def child_tree_deepth(node_dict,nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth += 1
    deepth += node_dict[nid].child_h
    return deepth


def CompressDelta(node1,p_node):
    a = node1.child_cut
    v1 = node1.vol + 1
    v2 = p_node.vol + 1
    return a * math.log2(v2/v1)


def CombineDelta(node1, node2, cut_v, g_vol):
    v1 = node1.vol + 1
    v2 = node2.vol + 1
    g1 = node1.g + 1
    g2 = node2.g + 1
    v12 = v1 + v2
    return ((v1 - g1) * math.log2(v12/v1) + (v2 - g2) * math.log2(v12/v2) - 2 * cut_v * math.log2(g_vol/v12)) / g_vol



class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, children:set = None, parent = None, child_h = 0, child_cut = 0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children if children is not None else set()
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h #不包括该节点的子树高度
        self.child_cut = child_cut
        self.center = None
        self.depth = 0
        self.E_increase = 0
        self.A_increase = 0
        self.cur_E = 0


    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

class PartitionTree():

    def __init__(self,adj_matrix):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(adj_matrix)
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()
        self.node_color = {}
        self.max_depth = 0

    def assign_colors_to_non_leaf_nodes(self,max_depth):
        """
        为每个非叶子节点分配颜色
        """
        # 使用Spectral色标
        cmap = plt.get_cmap("Spectral")

        # 获取所有非叶子节点（即有子节点的节点）
        non_leaf_nodes = [node_id for node_id, node in self.tree_node.items() if node.depth < self.max_depth]

        # 遍历非叶子节点，为它们分配颜色
        for node_id in non_leaf_nodes:
            # 根据节点的深度为其分配颜色
            depth = self.calculate_depth(node_id)
            color = cmap(depth / max_depth)  # 假设树的深度最多为10
            self.node_color[node_id] = mcolors.rgb2hex(color[:3])  # 只使用RGB值，去掉Alpha通道

    def calculate_depth(self, node_id):
        """
        计算节点的深度
        """
        depth = 0
        node = self.tree_node[node_id]
        while node.parent is not None:
            node = self.tree_node[node.parent]
            depth += 1
        return depth

    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = self.node_vol[vertex]
            leaf_node = PartitionTreeNode(ID=ID, partition=[vertex], g = v, vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)


    def build_sub_leaves(self,node_list,p_vol):
        subgraph_node_dict = {}
        ori_ent = 0
        for vertex in node_list:
            ori_ent += -(self.tree_node[vertex].g / self.VOL)\
                       * math.log2((self.tree_node[vertex].vol+1)/(p_vol+1))
            sub_n = set()
            vol = 0
            for vertex_n in node_list:
                c = self.adj_matrix[vertex,vertex_n]
                if c != 0:
                    vol += c
                    sub_n.add(vertex_n)
            sub_leaf = PartitionTreeNode(ID=vertex,partition=[vertex],g=vol,vol=vol)
            subgraph_node_dict[vertex] = sub_leaf
            self.adj_table[vertex] = sub_n

        return subgraph_node_dict,ori_ent

    def build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            ori_en += -(node.g / g_vol) * math.log2((node.vol+1)/g_vol)
            new_n = set()
            for nei in self.adj_table[node_id]:
                if nei in root_child:
                    new_n.add(nei)
            self.adj_table[node_id] = new_n

            new_node = PartitionTreeNode(ID=node_id,partition=node.partition,vol=node.vol,g = node.g,children=node.children)
            subgraph_node_dict[node_id] = new_node

        return subgraph_node_dict, ori_en


    def entropy(self,node_dict = None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id,node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol + 1
                node_g = node.g
                node_p_vol = node_p.vol + 1
                ent += - (node_g / self.VOL) * math.log2(node_vol/node_p_vol)
        return ent


    def __build_k_tree(self,g_vol,nodes_dict:dict,k = None,):
        min_heap = []
        cmp_heap = []
        nodes_ids = nodes_dict.keys()
        new_id = None
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    if len(n1.partition) == 1 and len(n2.partition) == 1:
                        cut_v = self.adj_matrix[n1.partition[0],n2.partition[0]]
                    else:
                        cut_v = cut_volume(self.adj_matrix,p1 = np.array(n1.partition),p2=np.array(n2.partition))
                    diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
                    heapq.heappush(min_heap, (diff, i, j, cut_v))
        unmerged_count = len(nodes_ids)

        while unmerged_count > 1:
            if len(min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = next(self.id_g)
            merge(new_id, id1, id2, cut_v, nodes_dict)
            self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            #compress delta
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap,[CompressDelta(nodes_dict[id1],nodes_dict[new_id]),id1,new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap,[CompressDelta(nodes_dict[id2],nodes_dict[new_id]),id2,new_id])
            unmerged_count -= 1

            for ID in self.adj_table[new_id]:
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    cut_v = cut_volume(self.adj_matrix,np.array(n1.partition), np.array(n2.partition))

                    new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
                    heapq.heappush(min_heap, (new_diff, ID, new_id, cut_v))
        root = new_id

        if unmerged_count > 1:
            #combine solitary node
            # print('processing solitary node')
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max([nodes_dict[i].child_h for i in unmerged_nodes]) + 1

            new_id = next(self.id_g)
            new_node = PartitionTreeNode(ID=new_id,partition=list(nodes_ids),children=unmerged_nodes,
                                         vol=g_vol,g = 0,child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [CompressDelta(nodes_dict[i], nodes_dict[new_id]), i, new_id])
            root = new_id

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                # print(nodes_dict[root].child_h,len(nodes_dict),len(cmp_heap))
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[e[2]])
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[p_id])
                heapq.heapify(cmp_heap)
        return root

    def calculate_entropy_increase(self):
        # 计算树中每个非叶子节点展开为子节点后图中的结构熵增加了多少
        entropy_increases = {}

        for node_id, node in self.tree_node.items():
            # 检查当前节点是否是非叶子节点
            self.tree_node[node_id].depth = self.calculate_depth(node_id)
            if self.tree_node[node_id].depth <  self.max_depth:  # 非叶子节点
                # 展开该节点，获取展开后的子节点列表
                new_children = [self.tree_node[i] for i in node.children]

                # 计算展开后所有子节点的熵之和
                expanded_entropy = sum(
                    -(child.g / self.VOL) * math.log2((child.vol + 1) / (self.tree_node[child.parent].vol + 1))
                    for child in new_children
                )

                # 记录该节点的熵增加量（就是子节点熵之和）
                entropy_increases[node_id] = expanded_entropy

        return entropy_increases

    def calculate_accumulated_entropy(self):
        """
        计算节点的累积熵增：包括节点自身的熵增和所有祖宗节点的熵增。
        """
        entropy_acc = {}
        for node_id, node in self.tree_node.items():
            # 检查当前节点是否是非叶子节点
            self.tree_node[node_id].depth = self.calculate_depth(node_id)
            if self.tree_node[node_id].depth < self.max_depth:  # 非叶子节点
                accumulated_entropy = node.E_increase
                current_node = node
                while current_node.parent is not None:
                    current_node = self.tree_node[current_node.parent]
                    accumulated_entropy += current_node.E_increase
                entropy_acc[node_id] = accumulated_entropy
                self.tree_node[node_id].A_increase = accumulated_entropy
        return entropy_acc

    def check_balance(self,node_dict,root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self.single_up(node_dict,c)

    def single_up(self,node_dict,node_id):
        new_id = next(self.id_g)
        p_id = node_dict[node_id].parent
        grow_node = PartitionTreeNode(ID=new_id, partition=node_dict[node_id].partition, parent=p_id,
                                      children={node_id}, vol=node_dict[node_id].vol, g=node_dict[node_id].g)
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1
        self.adj_table[new_id] = self.adj_table[node_id]
        adj_list = list(self.adj_table[node_id])  # 将集合转为列表进行迭代
        for i in adj_list:
            self.adj_table[i].add(new_id)

        # 修改完后，确保移除原子节点的邻接表
        self.adj_table[node_id] = set()  # 或者直接删除该节点的邻接表，如果需要



    def root_down_delta(self):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0 , None , None
        subgraph_node_dict, ori_entropy = self.build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self.__build_k_tree(g_vol=g_vol,nodes_dict=subgraph_node_dict,k=2)
        self.check_balance(subgraph_node_dict,new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self,sub_node_dict,sub_root_id,node_id):
        ent = 0
        for sub_node_id in LayerFirst(sub_node_dict,sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g

            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2((node.vol+1)/(node_p.vol+1))
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2((node.vol+1)/(node_p.vol+1))
        return ent

    def leaf_up(self):
        h1_id = set()
        h1_new_child_tree = {}
        id_mapping = {}
        for l in self.leaves:
            p = self.tree_node[l].parent
            h1_id.add(p)
        delta = 0
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if len(sub_nodes) == 1:
                id_mapping[node_id] = None
            if len(sub_nodes) == 2:
                id_mapping[node_id] = None
            if len(sub_nodes) >= 3:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict,ori_ent = self.build_sub_leaves(sub_nodes,candidate_node.vol)
                sub_root = self.__build_k_tree(g_vol=sub_g_vol,nodes_dict=subgraph_node_dict,k = 2)
                self.check_balance(subgraph_node_dict,sub_root)
                new_ent = self.leaf_up_entropy(subgraph_node_dict,sub_root,node_id)
                delta += (ori_ent - new_ent)
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta,id_mapping,h1_new_child_tree

    def leaf_up_update(self,id_mapping,leaf_up_dict):
        for node_id,h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self.single_up(self.tree_node,i)
            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    assert h1_c not in self.tree_node
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1


    def root_down_update(self, new_id , root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, k=2, mode='v2'):
        self.max_depth = k
        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=k)
        elif mode == 'v2':
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=2)
            root=self.tree_node[self.root_id]
            self.check_balance(self.tree_node,self.root_id)

            if self.tree_node[self.root_id].child_h < 2:
                self.tree_node[self.root_id].child_h = 2

            flag = 0
            root = self.tree_node[self.root_id]
            while self.tree_node[self.root_id].child_h < k:
                if flag == 0:
                    leaf_up_delta,id_mapping,leaf_up_dict = self.leaf_up()
                    root_down_delta, new_id , root_down_dict = self.root_down_delta()

                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                elif flag == 2:
                    root_down_delta, new_id , root_down_dict = self.root_down_delta()
                else:
                    raise ValueError
                # print(leaf_up_delta, root_down_delta)
                if leaf_up_delta < root_down_delta:
                    # root down update and recompute root down delta
                    flag = 2
                    self.root_down_update(new_id,root_down_dict)

                else:
                    # leaf up update
                    flag = 1
                    # print(self.tree_node[self.root_id].child_h)
                    self.leaf_up_update(id_mapping,leaf_up_dict)
                    # print(self.tree_node[self.root_id].child_h)


                    # update root down leave nodes' children
                    if root_down_delta != 0:
                        for root_down_id, root_down_node in root_down_dict.items():
                            if root_down_node.child_h == 0:
                                root_down_node.children = self.tree_node[root_down_id].children
        count = 0
        for _ in LayerFirst(self.tree_node, self.root_id):
            count += 1
        assert len(self.tree_node) == count

    def __build_new_tree(self, g_vol, nodes_dict: dict, k: int):
        min_heap = []
        nodes_ids = list(nodes_dict.keys())

        # 初始化堆（合并熵增最小的节点对）
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:
                    n1, n2 = nodes_dict[i], nodes_dict[j]
                    cut_v = self.cut_volume(n1.partition, n2.partition)
                    delta_H = CombineDelta(n1, n2, cut_v, g_vol)
                    heapq.heappush(min_heap, (delta_H, i, j, cut_v))

        unmerged_count = len(nodes_ids)
        new_nodes = {nid: copy.deepcopy(node) for nid, node in nodes_dict.items()}  # 深拷贝避免污染原数据

        # 合并到 k 个子节点，且强制子节点为叶子（child_h=0）
        while unmerged_count > k and min_heap:
            delta_H, id1, id2, cut_v = heapq.heappop(min_heap)
            if new_nodes[id1].merged or new_nodes[id2].merged:
                continue

            # 生成新节点（高度=1，子节点为原子节点且强制设为叶子）
            new_id = next(self.id_g)
            merged_node = PartitionTreeNode(
                ID=new_id,
                partition=new_nodes[id1].partition + new_nodes[id2].partition,
                parent=None,
                children={id1, id2},
                vol=new_nodes[id1].vol + new_nodes[id2].vol,
                g=new_nodes[id1].g + new_nodes[id2].g + cut_v,
                child_h=1  # 新节点高度为1
            )
            # 原子节点标记为叶子（child_h=0）
            new_nodes[id1].child_h = 0
            new_nodes[id2].child_h = 0
            new_nodes[id1].parent = new_id
            new_nodes[id2].parent = new_id

            new_nodes[new_id] = merged_node
            new_nodes[id1].merged = True
            new_nodes[id2].merged = True
            unmerged_count -= 1

        # 返回未合并的节点（即k个子节点）
        return [n for n in new_nodes.values() if not n.merged]

    def split_non_leaf_node_optimized(self, node_id, k=2):
        node = self.tree_node[node_id]
        # 提取子树所有叶子节点
        leaf_nodes = self._extract_subtree_leaves(node_id)

        # 构建子图参数
        sub_adj, sub_vol, node_mapping = self.build_subgraph(leaf_nodes)
        sub_nodes = {nid: PartitionTreeNode(
            ID=nid,
            partition=node.partition,
            vol=sub_vol[nid],
            g=0,
            child_h=0  # 初始为叶子节点
        ) for nid in node_mapping}

        # 合并到k个子节点
        new_children = self.__build_new_tree(g_vol=sum(sub_vol.values()), nodes_dict=sub_nodes, k=k)

        # 更新原树结构
        for child in new_children:
            child.parent = node_id
            child.child_h = 1  # 新子节点高度为1
            self.tree_node[child.ID] = child
            # 更新原子节点的父节点
            for leaf_id in child.children:
                self.tree_node[leaf_id].parent = child.ID
        # 原节点更新子节点和高度
        node.children = {c.ID for c in new_children}
        node.child_h = 2  # 原节点高度提升
        return new_children


    def _extract_subtree_leaves(self, node_id):
        """提取子树的所有叶子节点"""
        leaves = []
        stack = [self.tree_node[node_id]]
        while stack:
            current = stack.pop()
            if current.child_h == 0:
                leaves.append(current)
            else:
                stack.extend([self.tree_node[cid] for cid in current.children])
        return leaves

    def build_subgraph(self, leaf_nodes):
        """构建子图的邻接矩阵和体积"""
        sub_nodes = [node.partition for node in leaf_nodes]
        sub_adj = self.adj_matrix[np.ix_(sub_nodes, sub_nodes)]
        sub_vol = {node.ID: node.vol for node in leaf_nodes}
        return sub_adj, sub_vol, {idx: node.ID for idx, node in enumerate(leaf_nodes)}

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

def download_citeseer_to_local():
    """
    下载CiteSeer数据集到本地文件夹
    返回：数据集文件路径
    """
    base_url = "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"
    data_dir = "citeseer_data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 下载文件
    tgz_path = f"{data_dir}/citeseer.tgz"
    if not os.path.exists(tgz_path):
        print("正在下载CiteSeer数据集...")
        urllib.request.urlretrieve(base_url, tgz_path)
    else:
        print("CiteSeer压缩文件已存在，跳过下载")

    # 解压文件
    extract_dir = f"{data_dir}/citeseer"
    if not os.path.exists(extract_dir):
        print("正在解压文件...")
        tar = tarfile.open(tgz_path)
        tar.extractall(path=data_dir)
        tar.close()
    else:
        print("CiteSeer文件已解压，跳过解压步骤")

    # 检查文件是否存在
    cites_path = f"{extract_dir}/citeseer.cites"
    content_path = f"{extract_dir}/citeseer.content"

    if not os.path.exists(cites_path) or not os.path.exists(content_path):
        print(f"查找文件位置: {os.listdir(extract_dir)}")
        raise FileNotFoundError(f"无法在 {extract_dir} 中找到预期的CiteSeer文件")

    print(f"CiteSeer数据集已成功下载并解压到 {extract_dir}")
    return {"cites_path": cites_path, "content_path": content_path}


def load_citeseer_from_local(file_paths=None):
    """
    从本地文件加载CiteSeer数据集并创建NetworkX图
    参数：
        file_paths: 包含cites_path和content_path的字典
    返回：
        NetworkX图对象
    """
    if file_paths is None:
        # 使用默认路径
        data_dir = "citeseer_data"
        extract_dir = f"{data_dir}/citeseer"
        file_paths = {
            "cites_path": f"{extract_dir}/citeseer.cites",
            "content_path": f"{extract_dir}/citeseer.content"
        }

    # 检查文件是否存在
    if not os.path.exists(file_paths["cites_path"]) or not os.path.exists(file_paths["content_path"]):
        raise FileNotFoundError(f"找不到CiteSeer数据文件，请先运行download_citeseer_to_local()")

    # 读取引用数据
    cites = pd.read_csv(file_paths["cites_path"], sep='\t', header=None, names=['cited', 'citing'])

    # 读取内容数据
    content = pd.read_csv(file_paths["content_path"], sep='\t', header=None)

    # 显示数据结构，帮助调试
    print(f"内容数据形状: {content.shape}")
    num_columns = content.shape[1]
    print(f"总列数: {num_columns}")

    # 创建图
    G = nx.DiGraph()

    # 添加节点及其属性
    for i, row in content.iterrows():
        node_id = row[0]  # 第一列是节点ID
        features = row[1:num_columns - 1].values  # 中间列是特征
        label = row[num_columns - 1]  # 最后一列是标签
        G.add_node(node_id, features=features, label=label)

    # 添加边
    for i, row in cites.iterrows():
        G.add_edge(row['cited'], row['citing'])

    print(f"创建了包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边的图")
    return G
if __name__ == "__main__":
    file_paths = download_citeseer_to_local()

    # 第二步：从本地加载数据并创建图
    G = load_citeseer_from_local(file_paths)
    adj_sparse = nx.adjacency_matrix(G)
    adj_matrix = adj_sparse.toarray().astype(np.float32)

    # 创建 PartitionTree 实例
    y = PartitionTree(adj_matrix)

    # 构建分区树
    y.build_coding_tree(k=2, mode='v2')
    entropy_increases = y.calculate_entropy_increase()
    # 打印结果
    print(y.root_id)
    for k, v in y.tree_node.items():
        print(f"Node ID: {k} - {v.gatherAttrs()}")
