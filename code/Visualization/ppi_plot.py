import os.path
from random import sample
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import nx_to_g
from networkx.convert_matrix import from_pandas_edgelist, from_pandas_adjacency
from networkx.algorithms.shortest_paths.generic import shortest_path
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import graph_tool.all as gt
from functools import reduce

# Class to draw circle markers in the legend.
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Circle(xy=center, radius=80+width + height + ydescent + xdescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

# Create a graph from a csv file of PPI edge list.
def ppi_from_file(ppi_path):
    ppi = pd.read_csv(ppi_path, header=0, index_col=None).loc[:, ["Symbol_A", "Symbol_B"]]

    ppi.columns = ["source", "target"]
    ppi.index = range(len(ppi.index))
    ppi.dropna(inplace=True)
    G = from_pandas_edgelist(ppi, source="source", target="target")
    return G

# Compute and draw shortest paths between the signature genes and the LCC.
# If we have several signatures over different seeds, aggregate them.
def draw_paths_lcc(G, signature, lcc, name):
    G.remove_edges_from(nx.selfloop_edges(G))
    paths = pd.DataFrame(data=np.zeros((len(G.nodes), len(G.nodes))), columns=G.nodes, index=G.nodes)
    total = 0
    lcc = [idx for idx in lcc if idx in G.nodes]
    G_lcc = G.subgraph(lcc)
    # Keep the LCC of the disease genes.
    lcc = list(max(nx.connected_components(G_lcc), key=lambda x: len(x)))
    # Compute the shortest paths
    divide = 0
    if not os.path.exists(name + "_" + str(len(signature) - 1) + ".csv"):
        for seed in range(len(signature)):
            for gene1 in signature[seed]:
                if gene1 in lcc:
                    paths.loc[gene1, gene1] = 1.
                    continue
                path = []
                for gene2 in lcc:
                    try:
                        aux = shortest_path(G, source=gene1, target=gene2)
                    except Exception:
                        continue
                    if len(path) == 0 or len(path) > len(aux):
                        path = aux
                # Count the total length of the shortest paths
                total += len(path) - 1
                divide += 1
                for i in range(len(path)):
                    if i + 1 < len(path):
                        paths.loc[path[i + 1], path[i]] += 1.
                        paths.loc[path[i], path[i + 1]] += 1.
        # Only keep genes from the signature, the LCC and the intermediaries in the shortest paths.
        lcc = [gene for gene in lcc if gene in paths[0].columns]
        keep_lcc = [idx for idx in lcc if sum(paths.loc[idx]) > 0]
        # Keep the shortest paths in the LCC
        for gene_idx1 in range(len(keep_lcc)):
            gene1 = keep_lcc[gene_idx1]
            for gene_idx2 in range(gene_idx1, len(keep_lcc)):
                gene2 = keep_lcc[gene_idx2]
                try:
                    aux = shortest_path(G_lcc, source=gene1, target=gene2)
                except:
                    continue
                if len(aux) > 0:
                    for seed in range(len(signature)):
                        for i in range(len(aux)):
                            if i + 1 < len(aux):
                                paths.loc[aux[i + 1], aux[i]] += 1.
                                paths.loc[aux[i], aux[i + 1]] += 1.
        keep = [idx for idx in G.nodes if sum(paths.loc[idx]) > 0]

        paths = paths.loc[
                set([i for i in keep if i in G.nodes]), set([i for i in keep if i in G.nodes])]

        paths.to_csv(name + "_paths.csv")
    else:
        paths = pd.read_csv(name + "_paths.csv", index_col=0)
    # Create a graph from the paths and give the nodes a property to color them.
    F = nx.from_pandas_adjacency(paths)
    for node in F.nodes:
        F.nodes[node]['part_of'] = []
    # Give the nodes their color according to which  seed of signature they belong to, if they belong to the LCC or the intermediaries
    for node in F.nodes():
        if any(node in feature for feature in signature):
            for seed in range(len(signature)):
                if node in signature[seed]:
                    F.nodes[node]['part_of'].append(seed + 1 + 3)
        if len(F.nodes[node]['part_of']) == 0:
            F.nodes[node]['part_of'].append(0)
    pos = nx.spring_layout(F, center=(0, 0))

    fig, ax = plt.subplots(figsize=(80, 80))
    # Define the labels for the legend
    labels = ["Seed " + seed for seed in seeds] + [hall for hall in hallmarks] + ["Path Intermediates"]
    # Define the colormap
    cmap = plt.cm.viridis
    colors = [cmap(4 / 10), cmap(5 / 10), cmap(6 / 10), cmap(7 / 10), cmap(10/10), 'w']
    for idx in range(len(labels)):
        if idx != len(labels) - 1:
            ax.plot([0], [0], color=colors[idx], label=labels[idx], markeredgecolor=colors[idx], marker='o')
        else:
            ax.plot([0], [0], color=colors[idx], label=labels[idx], markeredgecolor=cmap(0), marker='o')
    pos = nx.rescale_layout_dict(pos, scale=8.)
    nx.draw_networkx_edges(F, pos=pos)
    # Set the colors of the nodes, each node is associated to a pie chart
    for node in F.nodes():
        part_of = list(set(F.nodes[node]['part_of']))
        if part_of == [0]:
            w = plt.pie(
                [1] * len(part_of),
                center=pos[node],
                colors="w",
                wedgeprops={"edgecolor":cmap(0)},
                radius=0.15,
            )
        else:
            print([q / 10 for q in part_of])
            w = plt.pie(
                [1] * len(part_of),
                center=pos[node],
                colors=[cmap(q / 10) for q in part_of],
                radius=0.15,
            )
        if any(node in feature for feature in signature):
            ax.annotate(node, (pos[node][0] + .03, pos[node][1] + .03), fontsize=50, color="#ADADC9")
    plt.axis('off')
    fig.set_facecolor('w')
    plt.xlim(-10, 10)
    plt.ylim(-10, 8)
    leg = plt.legend(prop={'size': 50}, loc='lower right', handler_map={mpatches.Circle: HandlerCircle()})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(0.0)
        legobj.set_marker('o')
        legobj.set_markersize(40)
    plt.savefig(name + ".svg")
    # Print the average distance to the LCC
    print(name + ".svg", total / divide)
    # Print the legend in a separate file
    handles, labels = ax.get_legend_handles_labels()
    leg = plt.legend(prop={'size': 80}, handler_map={mpatches.Circle: HandlerCircle()})
    figlegend = plt.figure(figsize=(10, 10))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(0.0)
    figlegend.legend(handles, labels, loc='upper center')
    plt.savefig(name + "_legend.svg")



def draw_paths_signature(G, signature, name):
    seeds = ["10", "82", "94", "118"]
    # Trim the graph by removing self loops
    G.remove_edges_from(nx.selfloop_edges(G))
    total = 0
    divide = 0
    if not os.path.exists(name + ".csv"):
        paths = pd.DataFrame(data=np.zeros((len(G.nodes), len(G.nodes))), columns=G.nodes, index=G.nodes)
        # Compute the shortest paths between each pair of genes of the differnet signatures.
        for gene1 in set([g for sig in signature for g in sig]):
            divide += 1
            for seed2 in range(len(signature)):
                signature2 = signature[seed2]
                if gene1 in signature2:
                    continue
                path = []
                for gene2 in signature2:
                    try:
                        aux = shortest_path(G, source=gene1, target=gene2)
                    except:
                        continue
                    if len(path) == 0 or len(path) > len(aux):
                        path = aux
                total += len(path) - 1
                for i in range(len(path)):
                    if i + 1 < len(path):
                        paths.loc[path[i + 1], path[i]] += 1.
                        paths.loc[path[i], path[i + 1]] += 1.
        # Only keep the genes of the signature and the intermediaries
        keep = [idx for idx in paths.columns if sum(paths[idx]) > 0]
        paths = paths.loc[
            set([i for i in keep if i in G.nodes]), set([i for i in keep if i in G.nodes])]
        paths.to_csv(name + ".csv")
    else:
        paths = pd.read_csv(name + ".csv", index_col=0)

    F = nx.from_pandas_adjacency(paths)
    for node in F.nodes:
        F.nodes[node]['part_of'] = []

    # Compute a spring layout
    if not os.path.exists(name + "_position_spring.npy"):
        pos = nx.spring_layout(F, center=(0, 0))
        np.save(name + "_position_spring.npy", pos)
    else:
        pos = np.load(name + "_position_spring.npy", allow_pickle=True).item()
    # Give the nodes their color according to which  seed of signature they belong to, if they belong to the LCC or the intermediaries
    for node in F.nodes():
        if any(node in feature for feature in signature):
            for seed in range(len(signature)):
                if node in signature[seed]:
                    F.nodes[node]['part_of'].append(seed + 1 + 3)
        if len(F.nodes[node]['part_of']) == 0:
            F.nodes[node]['part_of'].append(0)

    fig, ax = plt.subplots(figsize=(80, 80))

    # Define the color map for the nodes
    cmap = plt.cm.viridis
    labels = ["Seed " + seeds[0], "Seed " + seeds[1], "Seed " + seeds[2], "Seed " + seeds[3],
              "Path Intermediates"]
    colors = [cmap(4 / 10), cmap(5 / 10), cmap(6 / 10), cmap(7 / 10), "w"]
    for idx in range(len(labels)):
        ax.plot([0], [0], color=colors[idx], label=labels[idx], marker='o')
    pos = nx.rescale_layout_dict(pos, scale=1.1)
    nx.draw_networkx_edges(F, pos=pos, )
    # Color the nodes according to the different groups they are part of, a node is drawn as a pie chart.
    for node in F.nodes():
        part_of = list(set(F.nodes[node]['part_of']))
        if part_of != [0]:
            w = plt.pie(
                [1] * len(part_of),
                center=pos[node],
                colors=[cmap(q / (len(signature) + 6)) for q in part_of],
                radius=0.04,

            )
        else:
            w = plt.pie(
                [1] * len(part_of),
                center=pos[node],
                colors=["w"],
                wedgeprops={"edgecolor": cmap(0)},
                radius=0.04,

            )
        if any(node in feature for feature in signature):
            ax.annotate(node, (pos[node][0] + .02, pos[node][1] + .02), fontsize=50, color="#ADADC9")

    plt.axis('off')
    fig.set_facecolor('w')

    plt.savefig(name + ".svg")
    # Print the average distance between the signatures
    if divide != 0:
        print(name + ".svg", total / divide, divide)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)


if __name__ == "__main__":
    import sys

    signature_path = sys.argv[1]
    ppi_path = sys.argv[2]
    disease_genes_path = sys.argv[3]
    random = int(sys.argv[4])

    G_ppi = ppi_from_file_full(ppi_path)
    disease_genes = pd.read_csv(disease_genes_path)
    if random==0:
        signature = list(np.loadtxt(signature_path + ".txt", delimiter="\n", dtype="str"))
    else:
        signature = [genes[idx] for idx in sample(range(0, len(G_ppi.nodes)), random)]

    draw_paths_signature(G_ppi, signature, signature_path)
    draw_paths_lcc(G_ppi, signature, module, signature_path)

