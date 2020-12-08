import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d import proj3d
from tabulate import tabulate

from config import *


def specalbum(list_of_sne, labels=[], title=''):
    if not isinstance(list_of_sne, list):
        list_of_sne = [list_of_sne]
    if not labels:
        labels = ['' for _ in list_of_sne]
    labels_updated = labels.copy()

    def updatelabels(i_, t):
        labels_updated[i_] = labels[i_] + ' at ' + str(round(t, 1)) + ' ' + 'd'

    allflux = np.concatenate([sn.flux.flatten() for sn in list_of_sne])
    # plt.plot(range(len(allflux)),allflux)
    # yl=plt.ylim()
    # plt.close()

    ax = plt.axes()
    fig = ax.get_figure()
    lines = []
    # plt.subplots_adjust(bottom=0.2)
    plt.title(title)
    plt.xlabel(lambstr)
    plt.ylabel(r'$f_{\lambda}$, scaled')
    for i, sn in enumerate(list_of_sne):
        l, = ax.plot(sn.lamb, sn.flux[0, :], '-')
        lines.append(l)
        updatelabels(i, sn.time[0])
    plt.legend(labels_updated)
    plt.grid()
    # plt.ylim(yl)

    # plt.xlim(llim)

    plt.ylim([np.nanmin(allflux), np.nanmax(allflux)])

    class Index(object):
        ind = 0

        def press(self, event):
            if event.key == 'right':
                self.ind += 1
                self.plot()
            elif event.key == 'left':
                self.ind -= 1
                self.plot()

        def plot(self):
            for i, sn in enumerate(list_of_sne):
                self.ind = max(min(self.ind, len(sn.time) - 1), 0)
                lines[i].set_ydata(sn.flux[self.ind, :])
                updatelabels(i, sn.time[self.ind])
            plt.legend(labels_updated)
            plt.draw()

    callback = Index()
    fig.canvas.mpl_connect('key_press_event', callback.press)
    plt.show()


def visualizemetrics(snlist):
    embs = [sn.features for sn in snlist]
    smallest = []
    for i, emb in enumerate(embs):
        a = emb.copy()
        a[i] = np.inf
        smallest += [np.argmin(a)]
    embs = list(zip(*embs))

    table = []
    headers = ['Tmpl. \\ Trgt.'] + [sn.name for sn in snlist]

    for i, sn in enumerate(snlist):
        line = [sn.name]
        for j, x in enumerate(embs[i]):
            cell = str(round(x, 2))
            if i == smallest[j]:
                cell = '//' + cell + '//'
            line.append(cell)

        table.append(line)
    tablestr = tabulate(table, headers=headers)
    # timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    # with open(os.path.join(MAIN_DIR, 'metricstable_' + timestamp), 'w') as f:
    #     f.write(tablestr)
    print(tablestr)


def visualizeclustering(clustering, txtsavepath=None):
    def cellformat(sn, sil):
        return sn.name + ', ' + str(round(sil, 2))

    def headerformat(std):
        return 'sigma = ' + str(round(std, 3))

    lsts = [[] for _ in clustering.clusters]
    for sn, sil in sorted(zip(clustering.snlist, clustering.sil_samples()), key=lambda x: -x[1]):
        cell = cellformat(sn, sil)
        idx = next(i for i, cl in enumerate(clustering.clusters) if sn in cl)
        lsts[idx].append(cell)

    headers = [str(i + 1) + ') ' + headerformat(cl.std) for i, cl in enumerate(clustering.clusters)]

    tablestr = tabulate(dict(zip(headers, lsts)), headers='keys')
    print(tablestr)
    if txtsavepath is not None:
        with open(txtsavepath, 'w') as f:
            f.write(tablestr)


def annotate_onclick(scat, pcs, names):
    ax = scat.axes
    fig = ax.get_figure()

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def dataspace(point):
        if point.shape == (3,):
            x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        else:
            x2, y2 = point[0], point[1]
        return x2, y2

    def distance(point, event):
        # Project 3d data space to 2d data space
        x2, y2 = dataspace(point)
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))
        return np.sqrt((x3 - event.x) ** 2 + (y3 - event.y) ** 2)

    def update_annot(ind):
        x2, y2 = dataspace(pcs[ind, :])
        annot.xy = (x2, y2)
        text = names[ind]
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, _ = scat.contains(event)
            if cont:
                ind = min(range(pcs.shape[0]), key=lambda i: distance(pcs[i, :], event))
                update_annot(ind)
                annot.set_visible(True)
                fig.texts = [annot]
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", hover)


def myscatter(matrix, snlist, dims=None):
    if dims is None:
        dims = range(1, matrix.shape[1] + 1)
    matrix = matrix[:, np.array(dims) - 1]

    names = [sn.name for sn in snlist]
    fulltypes = [info_df['FullType'][nm] if not pd.isna(info_df['FullType'][nm]) else info_df['Type'][nm] for nm in
                 names]
    colors = [typ2color[typ] for typ in fulltypes]

    view_dim = matrix.shape[1]
    if view_dim not in [2, 3]:
        raise Exception('matrix must be [n_samples x 2 or 3]')

    ax = plt.axes(projection='3d' if view_dim == 3 else None)

    if view_dim == 2:
        scat = ax.scatter(matrix[:, 0], matrix[:, 1], c=colors, marker=marker, s=size)
        ax.set_xlabel('PC%s' % dims[0])
        ax.set_ylabel('PC%s' % dims[1])
    else:
        scat = ax.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], c=colors, marker=marker, s=size)
        ax.set_xlabel('PC%s' % dims[0])
        ax.set_ylabel('PC%s' % dims[1])
        ax.set_zlabel('PC%s' % dims[2])

    handles = [Line2D([0], [0], linewidth=0, color=c, marker=marker) for c in colors]
    by_label = dict(zip(fulltypes, handles))
    plt.legend(by_label.values(), by_label.keys())

    annotate_onclick(scat, matrix, names)

    plt.tight_layout()

    # if save:
    #     def rotate(angle):
    #         plt.gca().view_init(azim=angle)
    #     rot_animation = animation.FuncAnimation(plt.gcf(), rotate)
    #     with open('gif.html', 'w') as f:
    #         f.write(rot_animation.to_jshtml())

    plt.show()


def cornerplot(matrix, snlist, dims=None):
    if dims is None:
        dims = range(1, matrix.shape[1] + 1)
    matrix = matrix[:, np.array(dims) - 1]

    names = [sn.name for sn in snlist]
    fulltypes = [info_df['FullType'][nm] if not pd.isna(info_df['FullType'][nm]) else info_df['Type'][nm] for nm in
                 names]
    colors = [typ2color[typ] for typ in fulltypes]
    d = matrix.shape[1]
    fig, axs = plt.subplots(d, d)
    for i in range(d):
        for j in range(i, d):
            ax = axs[j, i]
            if i == j:
                ax.hist([matrix[np.array(fulltypes) == typ, i] for typ in typ2color.keys()], bins=20,
                        color=typ2color.values(), stacked=True)
                ax.set_yticks([tick for tick in ax.get_yticks() if tick > 0])
                ax.yaxis.tick_right()
                # ax.set_ylabel('#')
                # ax.yaxis.set_label_position("right")
            else:
                scat = ax.scatter(matrix[:, i], matrix[:, j], c=colors, marker=marker, s=size / d)
                annotate_onclick(scat, matrix[:, [i, j]], names)

            if j == d - 1:
                ax.set_xlabel('PC%s' % dims[i])
            else:
                ax.set_xticks([])

            if i != j:
                if i == 0:
                    ax.set_ylabel('PC%s' % dims[j])
                else:
                    ax.set_yticks([])

    for j in range(d):
        for i in range(j + 1, d):
            axs[j, i].remove()

    fig.align_xlabels()
    fig.align_ylabels()

    plt.subplots_adjust(wspace=0, hspace=0)

    handles = [Line2D([0], [0], linewidth=0, color=c, marker=marker) for c in colors]
    by_label = dict(zip(fulltypes, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=10)
    plt.show()
