import json
from collections import defaultdict
from glob import glob
from os import environ
from os.path import basename
from os.path import join

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import interp1d
from spectres import spectres
from tabulate import tabulate

"""
#############################
RESULTS-CRITICAL:
#############################
"""

withhostcorr = True
band_for_max = 'Bessell_B'

"""#############################"""

"""minor settings:"""

NUM_JOBS = 16

SNLIST_PATH = join(environ['HOME'], 'DropboxWIS/spectra_in_time/pickled_snlist')
PYCOCO_DIR = join(environ['HOME'], 'DropboxWIS/PyCoCo_templates')
SFDATA_DIR = join(environ['HOME'], 'DropboxWIS/superfit_data')

# visuals:
typ2color = {
    'Ia': 'black',
    'Ib': 'tab:blue',
    'Ic': 'tab:green',
    'Ic-BL': 'lime',
    'II': 'tab:red',
    'IIn': 'tab:orange',
    'IIb': 'tab:purple',
    '': 'tab:cyan',
    '': 'tab:brown',
    '': 'tab:pink',
    '': 'tab:olive',
}
typ2color = defaultdict(lambda: 'tab:gray', typ2color)

marker = 'o'
size = 80
alpha = 0.5

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['legend.title_fontsize'] = 10
# mpl.rcParams['legend.fancybox'] = True

lambstr = r'wavelength [$\AA$]'
fluxstr = r'$f_\lambda$ [erg s-1 cm-2 $\AA$-1]'
timestr = r'rest days - max $m_B$'

# not input:
if withhostcorr:
    PYCOCO_SED_PATH = join(PYCOCO_DIR, 'Templates_HostCorrected/pycoco_%s.SED')  # with host correction
    PYCOCO_FINAL_DIR = join(PYCOCO_DIR, 'Outputs/%s/FINAL_spectra_2dim')
else:
    PYCOCO_SED_PATH = join(PYCOCO_DIR, 'Templates_noHostCorr/pycoco_%s_noHostCorr.SED')  # w/o host correction
    PYCOCO_FINAL_DIR = join(PYCOCO_DIR, 'Outputs/%s/FINAL_spectra_2dim/HostNotCorr')

PYCOCO_INFO_PATH = join(PYCOCO_DIR, 'Inputs/SNe_Info/info.dat')

info_df = pd.read_csv(PYCOCO_INFO_PATH, delimiter=' ', index_col=0)

"""FUNCTIONS FOR READING PYCOCO:"""

band_wvl, band_throughput = np.genfromtxt(join(PYCOCO_DIR, 'Inputs/Filters/GeneralFilters/', band_for_max + '.dat'),
                                          unpack=True)  # for maxtime calculation


def timesincemax(time, lamb, flux, fluxerr):
    b = interp1d(band_wvl, band_throughput, fill_value=0, bounds_error=False)

    isnan = np.isnan(flux)
    flux_ = flux.copy()
    flux_[isnan] = 0
    synphots = np.average(flux_, axis=1, weights=b(lamb) * ~isnan)

    # synphots = simps(b(lamb) * flux, lamb)
    maxtime = time[np.nanargmax(synphots)]
    return maxtime


def read_pycoco_template_sed(sedpath, canonical_lamb):
    times, lambs, fluxes = zip(np.loadtxt(sedpath, comments='#', delimiter=' ', unpack=True))

    if isinstance(times, tuple):
        times = times[0]
    if isinstance(lambs, tuple):
        lambs = lambs[0]
    if isinstance(fluxes, tuple):
        fluxes = fluxes[0]

    dct = {}
    for t, lm, fl in zip(times, lambs, fluxes):
        if t in dct:
            dct[t].append((lm, fl))
        else:
            dct[t] = [(lm, fl)]

    fluxlst = []
    time = np.array(sorted(dct.keys()))
    for t in time:
        lsttup = sorted(dct[t], key=lambda tup: tup[0])
        lamb, flux = zip(*lsttup)

        flux = spectres(canonical_lamb, np.array(lamb), np.array(flux), None, np.nan, False)

        fluxlst.append(flux)

    flux_matrix = np.row_stack(fluxlst)
    return time, canonical_lamb, flux_matrix, flux_matrix * np.nan


def read_pycoco_template_outdir(pycoco_out_dir, canonical_lamb):
    time, fluxlst, fluxerrlst = [], [], []

    def path2t(pth):
        return float(basename(pth).split('_')[0])

    for path in sorted(glob(join(pycoco_out_dir, '*.txt')), key=path2t):
        t = path2t(path)
        lamb, flux, fluxerr = zip(np.loadtxt(path, comments='#', delimiter='\t', unpack=True))
        if isinstance(fluxerr, tuple):
            fluxerr = fluxerr[0]
        if isinstance(lamb, tuple):
            lamb = lamb[0]
        if isinstance(flux, tuple):
            flux = flux[0]

        lamb, flux, fluxerr = zip(*sorted(zip(lamb, flux, fluxerr), key=lambda tup: tup[0]))
        lamb, flux, fluxerr = np.array(lamb), np.array(flux), np.array(fluxerr)

        flux, fluxerr = spectres(canonical_lamb, lamb, flux, fluxerr, np.nan, False)

        time.append(t)
        fluxlst.append(flux)
        fluxerrlst.append(fluxerr)

    flux_matrix = np.row_stack(fluxlst)
    fluxerr_matrix = np.row_stack(fluxerrlst)
    time = np.array(time)
    return time, canonical_lamb, flux_matrix, fluxerr_matrix


""" plotting:"""


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


def sfagreement(names, fulltypes):
    fulltypesdict = dict(zip(names, fulltypes))
    with open(join(SFDATA_DIR, 'sftypes2type.json')) as f:
        sftype2type = json.load(f)
    notfound = set()
    sfresults = pd.read_csv(join(SFDATA_DIR, 'bestfits'), delimiter=',')
    counter = {nm: 0 for nm in names}
    matchcounter = {nm: 0 for nm in names}
    for row in sfresults.iterrows():
        counter[row[1]['Name']] += 1
        sftype = row[1]['SF_type'].split('/')[0].strip('"')
        if sftype not in sftype2type:
            notfound.add(sftype)
        elif sftype2type[sftype] == fulltypesdict[row[1]['Name']]:
            matchcounter[row[1]['Name']] += 1

    if notfound:
        for x in notfound:
            print(x)
        raise Exception('Fill the above in json')

    pcntagree = []
    for nm in names:
        pcntagree.append(matchcounter[nm] / counter[nm])

    return np.array(pcntagree)


def myscatter(matrix, snlist, dims=None, sfsize=False):
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

    pctgs = np.ones_like(colors)
    pctg2sz = lambda x: size * x
    if sfsize:
        pctgs = sfagreement(names, fulltypes)
        pctg2sz = lambda x: 2 * size * x + 10

    if view_dim == 2:
        scat = ax.scatter(matrix[:, 0], matrix[:, 1], c=colors, marker=marker, s=pctg2sz(pctgs), alpha=alpha)
        ax.set_xlabel('dim %s' % dims[0])
        ax.set_ylabel('dim %s' % dims[1])
    else:
        scat = ax.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], c=colors, marker=marker, s=pctg2sz(pctgs))
        ax.set_xlabel('dim %s' % dims[0])
        ax.set_ylabel('dim %s' % dims[1])
        ax.set_zlabel('dim %s' % dims[2])

    handles = [Line2D([0], [0], linewidth=0, color=c, marker=marker) for c in colors]
    by_label = dict(zip(fulltypes, handles))
    if sfsize:
        by_label.update({' ': Line2D([], [], linestyle='')})
        by_label.update({str(round(pctg * 100)) + '%': Line2D([0], [0], linewidth=0, color='k', marker=marker,
                                                              markersize=np.sqrt(pctg2sz(pctg))) for pctg in
                         [np.max(pctgs), 0.5, np.min(pctgs)]})
    plt.legend(by_label.values(), by_label.keys())

    annotate_onclick(scat, matrix, names)

    plt.tight_layout()

    # def rotate(angle):
    #     plt.gca().view_init(azim=angle*2)
    # rot_animation = animation.FuncAnimation(plt.gcf(), rotate)
    # with open('gif.html', 'w') as f:
    #     f.write(rot_animation.to_jshtml())

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
                scat = ax.scatter(matrix[:, i], matrix[:, j], c=colors, marker=marker, s=size / d, alpha=alpha)
                annotate_onclick(scat, matrix[:, [i, j]], names)

            if j == d - 1:
                ax.set_xlabel('dim %s' % dims[i])
            else:
                ax.set_xticks([])

            if i != j:
                if i == 0:
                    ax.set_ylabel('dim %s' % dims[j])
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


if __name__ == '__main__':
    print(info_df['Type'].value_counts())
