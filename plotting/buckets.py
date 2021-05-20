# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:28:43 2021

@author: james hazelden
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd    
from matplotlib.colors import LogNorm
from matplotlib.patches import ConnectionStyle
import networkx as nx


# 300 dpi: plain paper prints, 600: professional papers, 1200: posters, top quality.
dpi = 600

plt.figure(dpi=dpi)
buckets = pd.read_csv("../coupling_speed_test/HH_BUCKETS.txt", delimiter=',').to_numpy()
buckets = buckets[:, :-1]
plt.imshow(buckets+1,  norm=LogNorm(vmax=np.max(buckets)))
plt.colorbar()


# plt.figure(dpi=dpi)
# sm = np.sum(buckets, axis=1).reshape(-1, 1)
# sm_rep = np.repeat(sm, buckets.shape[1], 1)
# buckets_non_zero = buckets[sm_rep != 0].reshape(-1, buckets.shape[1])
# buckets_nm_row = buckets_non_zero / np.sum(buckets_non_zero, axis=1).reshape(-1, 1)
# plt.imshow(buckets_nm_row)
# plt.colorbar()

# plt.figure(dpi=dpi)
# sm2 = np.sum(buckets_nm_row, axis=0).reshape(1, -1)
# sm_rep2 = np.repeat(sm2, buckets_nm_row.shape[0], 0)
# buckets_non_zero2 = buckets_nm_row[sm_rep2 != 0].reshape(buckets_nm_row.shape[0], -1)
# buckets_nm_row2 = buckets_non_zero2 / np.sum(buckets_non_zero2, axis=0).reshape(1, -1)
# plt.imshow(buckets_nm_row2+1,  norm=LogNorm(vmax=np.max(buckets_nm_row2+1)))
# plt.colorbar()

# plt.figure(dpi=dpi)
# plt.hist(buckets[:])

# plt.figure(dpi=dpi)
# plt.plot(buckets_nm_row.transpose())
plt.figure(dpi=dpi)

for i in range(1):
    transmission = pd.read_csv("../coupling_speed_test/HH_WHITE_NET_TRANSMISSION_%d.txt" % i, delimiter=',').to_numpy()
    transmission = transmission[:, :-1]
    plt.figure(dpi=dpi)
    plt.hist(transmission[:])

    plt.figure(dpi=dpi)
    trans_norm=LogNorm(vmax=np.max(transmission+1))
    edge_color = []
    G = nx.Graph()
    for p in range(4):
        for q in range(4):
            G.add_edge(p, q, weight=transmission[p,q])
            edge_color.append((trans_norm(transmission[p,q]+1), 0, 0))
    edge_color = np.array(edge_color)
    edge_color = (edge_color - np.min(edge_color)) / (np.max(edge_color) - np.min(edge_color))
    
    # Compute position of nodes
    G = nx.path_graph(4)

    pos = nx.random_layout(G)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(
        G, pos,
        #edge_color = edge_color,
        connectionstyle="arc3,rad=2.0"
    )
 #   nx.draw_circular(G, node_size=10, edge_color = edge_color, connectionstyle="arc3,rad=2.0")
 
    plt.figure(dpi=dpi)
    div = transmission / buckets
    # Replace division by zero with zeros.
    div = np.where(div != div, 0.0, div)
    # plt.imshow(div+1, norm=LogNorm(vmax=np.max(div+1)))
    # plt.colorbar()
    
    if i == 0:
        plt.figure(dpi=dpi)
        e1 = np.abs(np.linalg.eigvals(buckets[:, :-1]))
        e1 = (e1 - np.min(e1)) / (np.max(e1) - np.min(e1))
        e2 = np.abs(np.linalg.eigvals(div[:, :-1]))
        e2 = (e2 - np.min(e2)) / (np.max(e2) - np.min(e2))
        plt.subplot(1,2,1)
        plt.plot(e1, color=[0.5, 0, 0], linewidth=1)
        plt.title('A', fontweight='bold', loc='left')
        plt.box(on=None)
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.plot(e2, color=[0.5, 0, 0], linewidth=1)
        plt.title('B', fontweight='bold', loc='left')
        plt.box(on=None)
        plt.axis('off')

    plt.figure(dpi=dpi)
    plt.subplot(131)
    plt.imshow(buckets+1, norm=LogNorm(vmax=np.max(buckets+1)), cmap='seismic')
    plt.title('A', loc='left', fontweight='bold')
    ax = plt.subplot(132)
    plt.imshow(transmission+1, norm=LogNorm(vmax=np.max(transmission+1)), cmap='seismic')
    ax.yaxis.set_visible(False) 
    plt.title('B', loc='left', fontweight='bold')
    ax = plt.subplot(133)
    plt.imshow(div+1, norm=LogNorm(vmax=np.max(div+1)), cmap='seismic')
    plt.title('C', loc='left', fontweight='bold')
    ax.yaxis.set_visible(False) 
    
