import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib import collections as mc
import matplotlib.font_manager as font_manager
import glob

mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf")
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True


def plot():

    # load csv
    data = np.loadtxt(open('dgg_results.csv', 'rb'), delimiter=',')

    original = data[:, ::3]
    dgg = data[:, 1::3]
    dgg_wl = data[:, 2::3]

    models = [original, dgg]
    models_by_dataset = [np.split(data, 5) for data in models]

    backbones = ['GCN', 'GraphSage', 'GAT', 'GCNII']
    datasets = ['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'Reddit']

    colors = ['#fb5607', '#ff006e', '#8338ec', '#ffbe0b']

    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(2, 3)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    axes = [ax00, ax01, ax02, ax10, ax11]

    for idx_data, ax in enumerate(axes):
        ax.set_title(datasets[idx_data])
        for idx_bb in range(len(backbones)):
            og = models_by_dataset[0][idx_data][idx_bb]
            ours = models_by_dataset[1][idx_data][idx_bb]
            ax.plot(
                np.arange(len(og)), og, label=backbones[idx_bb], ls='-',
                c=colors[idx_bb], lw=3
            )
            ax.plot(
                np.arange(len(ours)), ours, label=backbones[idx_bb], ls='--',
                c=colors[idx_bb], lw=3
            )
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Noisy edges (%)')


    # ax.legend(loc='lower right')
    fig.tight_layout()
    plt.show()

    # fig.suptitle(title)
    #
    # ax00.set_title("Cora")
    # ax01.set_title("DAGNet: Player B")
    # ax02.set_title("DAGNet: Player C")
    #
    # ax01.set_xlabel(
    #     "Fixed neighborhood for 3 different players. Here the graph is fixed and fully-connected. \n"
    #     "Each player is always connected to all other players."
    # )
    # ax11.set_xlabel(
    #     "Our neighborhood selection for 3 different players. \n "
    #     "Each player can select their neighborhood and its size.\n"
    #     "Dotted blue lines show redundant edges (i.e. neigbours that were not selected)."
    # )
    #
    # ax02.set_xlim([0, 500])
    # ax02.set_ylim([0, 500])
    # ax01.set_xlim([0, 500])
    # ax01.set_ylim([0, 500])
    #
    # ax00.grid(False)
    # ax01.grid(False)
    # ax02.grid(False)
    #
    # ax00.set_xticks([])
    # ax00.set_yticks([])
    # ax01.set_xticks([])
    # ax01.set_yticks([])


    plt.savefig(
        os.path.join(
            '/vol/research/sceneEvolution/models/dagnet/visualisations/supplementary_vis/vis',
            '{:04d}{:04d}.png'.format(idx_b, idx_t)
        )
    )
    fig.clf()
    plt.close(fig)

plot()