import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import os
import numpy as np
import matplotlib.colors as mc
import colorsys

# Lightens the given color by multiplying (1-luminosity) by the given amount.
# Input can be matplotlib color string, hex string, or RGB tuple.
def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# Draws bowplots for each feature selection method and each evaluation metric.
# Takes a dataframe for boxplot as defined in evaluation_ml.py and a path for the output file.
def draw_metrics(data_df, file_path):
    sns.set_style("whitegrid")
    ensemble_name = ["Majority", "MV", "WMV", "Densest", "k-Heavy", "W-k Heavy"]
    x_name = "Metric"
    methods = list(np.unique(data_df.loc[:, "Method"]))
    palette = iter([plt.cm.tab20(i) for i in range(len(methods))])
    dict_metrics = {"BA": "Balanced Accuracy", "WP": "Weighted Precision", "WR": "Weighted Recall", "WF": "Weighted F1"}

    data_df.loc[:, "Metric"] = [dict_metrics[i] for i in data_df.loc[:, "Metric"]]
    fig, ax = plt.subplots(figsize=(40, 12))

    sns.boxplot(x="Metric", y="Value",
                data=data_df,
                hue="Method",
                whiskerprops={'linestyle': '--'},
                hue_order=ensemble_name,
                ax=ax
                )
    # Plots every box in white with colored edges except for the two proposed feature selection methods that are filled
    # p counts the number of boxes drawn, colors are determined by the sleection method and repeated for every metric hence the modulo.
    p = 0
    for box in ax.patches:
        # print(box.__class__.__name__)
        if box.__class__.__name__ == 'PathPatch':
            box.set_edgecolor(palette[p % len(palette)])
            # All the boxes except the two last ones are in white, the two last ones are filled with a color slightly lighter than there edges.
            if p % len(palette) < len(palette) - 2:
                box.set_facecolor('white')
            else:
                box.set_facecolor(lighten_color(palette[p % len(palette)], 0.5))
            # Draws the boxes edges.
            for k in range(6 * p, 6 * (p + 1)):
                ax.lines[k].set_color(palette[p % len(palette)])
            p += 1
    # Removes the legend
    for legpatch in ax.get_legend().get_patches():
        col = legpatch.get_facecolor()
        legpatch.set_edgecolor(col)
        legpatch.set_facecolor('None')
    # Name the axes and set the font.
    plt.xlabel(x_name, size=26)
    ax.set(ylabel=None)
    ax.tick_params(labelsize=26)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(file_path + ".svg")

    # Save the legend in a separate file
    handles, labels = ax.get_legend_handles_labels()
    figlegend = plt.figure(figsize=(10, 10))
    for legobj in handles:
        legobj.set_linewidth(5.0)
    figlegend.legend(handles, labels, loc='upper center')
    plt.savefig(file_path + "_legend.svg")


# Draws lineplots for each feature selection method and each evaluation metric.
# Takes a dataframe for lineplot as defined in evaluation_ml.py and a path for the output file.
def lineplot(data_df, name, dataset):
    sns.set()
    sns.set_style("whitegrid")
    X = list(np.unique(data_df.loc[:, "Seed"]))
    X.sort()
    methods = list(np.unique(data_df.loc[:, "Method"]))
    palette = iter([plt.cm.tab20(i) for i in range(len(methods))])
    ensemble_name = ["Majority", "MV", "WMV", "Densest", "k-Heavy", "W-k Heavy"]
    order_separation = ["BA", "WP", "WR", "WF"]
    dict_metrics = {"BA": "Balanced Accuracy", "WP": "Weighted Precision", "WR": "Weighted Recall", "WF": "Weighted F1"}

    separations = list(np.unique(data_df.loc[:, "Metric"]))
    fig, axes = plt.subplots(1, len(separations), figsize=(40, 20))
    plt.rcParams.update({'font.size': 26})
    for metric in order_separation:
        index_metric = order_separation.index(metric)
        aux = data_df.loc[data_df.loc[:, "Metric"] == metric].reset_index()
        x_ticks = [X.index(i) for i in aux.loc[:, "Seed"]]
        ax = axes[index_metric]
        metric_name = dict_metrics[metric]

        sns.lineplot(ax=ax, x=x_ticks, y="Value", hue='Method',
                     hue_order=ensemble_name, palette=palette,
                     data=aux)
        ax.set_xticks(range(len(X)))
        ax.set_xticklabels(X)
        ax.set_xlabel("Seeds", size=26)

        ax.tick_params(labelsize=26)
        ax.set_ylabel(metric_name, size=30)
        ax.get_legend().remove()
        ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(file_path + ".svg")

    handles, labels = ax.get_legend_handles_labels()
    figlegend = plt.figure(figsize=(10, 10))
    for legobj in handles:
        legobj.set_linewidth(5.0)
    figlegend.legend(handles, labels, loc='upper center')
    plt.savefig(file_path + "_legend.svg")
