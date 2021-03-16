from analysis import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

rng = [[TIME[0], TIME[-1] + TIME[1] - TIME[0]], [LAMB[0], LAMB[-1] + LAMB[1] - LAMB[0]]]
xedges, yedges = np.linspace(rng[0][0], rng[0][1], 10), np.linspace(rng[1][0], rng[1][1], 50)


def examine_rf():
    exc, snlist, X, X_PC, m, scaler, dismat, build_dissimilarity_matrix, rand_f = train()
    indicator, n_nodes_ptr = rand_f.decision_path(X_PC)
    allpaths = [[] for _ in snlist]
    for i, clf in enumerate(rand_f.estimators_):
        tr_indicator = indicator[:, n_nodes_ptr[i]:n_nodes_ptr[i + 1]]
        for snidx, sn_treepaths in enumerate(allpaths):
            if clf.predict_proba(X_PC[snidx, :].reshape(1, -1))[:, 0] == 1:  # whether clf classifies this SN as real
                ftr_indices = [clf.tree_.feature[n] for n in range(tr_indicator.shape[1]) if
                               tr_indicator[snidx, n] != 0 and clf.tree_.feature[n] > 0]
                tind, lind = np.unravel_index(ftr_indices, (len(TIME), len(LAMB)))
                ftrs = np.column_stack((TIME[tind], LAMB[lind]))
                sn_treepaths.append(ftrs)
            else:  # if not a good tree for this SN, appned an empty path
                sn_treepaths.append(np.zeros((0, 2)))

    return snlist, allpaths


def draw_paths(allpaths, snlist, typs=None, num=4):
    if typs is not None:
        idxs = [i for i, sn in enumerate(snlist) if sn.type in typs]
        allpaths = [allpaths[i] for i in idxs]
        snlist = [snlist[i] for i in idxs]
    num_goodpaths = 0
    g = {}  # # a fully connected graph with nodes as features (binned).
    for sn, sn_treepaths in zip(snlist, allpaths):  # each edge weight is the number of paths containeing the two nodes
        for p in sn_treepaths:
            if not p.shape[0]:
                continue
            num_goodpaths += 1
            binx = np.digitize(p[:, 0], xedges) - 1
            biny = np.digitize(p[:, 1], yedges) - 1
            q = np.column_stack((binx, biny))
            nds = [(q[i, 0], q[i, 1]) for i in range(p.shape[0])]
            for i1 in range(len(nds)):
                for i2 in range(i1 + 1, len(nds)):
                    if not ((nds[i1], nds[i2]) in g or (nds[i2], nds[i1]) in g):
                        g[(nds[i1], nds[i2])] = 0
                    if (nds[i1], nds[i2]) in g:
                        g[(nds[i1], nds[i2])] += 1
                    else:
                        g[(nds[i2], nds[i1])] += 1

    sortededges = sorted(g.keys(), key=lambda e: -g[e])
    sortededges = sortededges[:num]
    g = {e: 100 * g[e] / num_goodpaths for e in sortededges}  # values are converted to % of all good decision paths
    maxweight, minweight = max(g.values()), min(g.values())

    ax_fl = plt.axes([0.1, 0.2, 0.8, 0.7])
    ax_c1 = plt.axes([0.1, 0.1, 0.2, 0.02])
    ax_c2 = plt.axes([0.7, 0.1, 0.2, 0.02])
    for e in sortededges:
        x, y = zip(e[0], e[1])
        x, y = np.array(x), np.array(y)
        x = (xedges[x] + xedges[x + 1]) / 2
        y = (yedges[y] + yedges[y + 1]) / 2
        ax_fl.plot(x, y, color=plt.cm.Reds((g[e] - minweight) / (maxweight - minweight)), alpha=1, marker='*')
    norm = mpl.colors.Normalize(vmin=minweight, vmax=maxweight)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax_c1, orientation='horizontal', pad=0.2,
                 label='% of valid SN paths with\n a feature from each 2d bin')

    c = plot_meanflux(snlist, TIME, LAMB, ax_fl)
    ax_fl.set_xlabel('days since explosion')
    ax_fl.set_ylabel(r'wavelength [$\AA$]')
    plt.colorbar(c, cax=ax_c2, orientation='horizontal', pad=0.2,
                 label='Flux rescaled')
    add_spectral_lines(ax_fl, cl='pink', allalong=True)

    plt.suptitle(', '.join(typs) if typs is not None else 'all')
    plt.show()


def hist_features(allpaths, snlist, typ, botpctl=98):
    selection = []
    snpathftrs = np.zeros((0, 2))
    for sn, sn_treepaths in zip(snlist, allpaths):
        for p in sn_treepaths:
            snpathftrs = np.row_stack((snpathftrs, p))
            selection.extend([sn.type == typ for _ in p])
    selection = np.array(selection)

    fig, (ax_fl, ax_hist) = plt.subplots(1, 2, sharex=True, sharey=True)

    h_all, _, _ = np.histogram2d(snpathftrs[:, 0], snpathftrs[:, 1], bins=[xedges, yedges])
    binsx = np.digitize(snpathftrs[:, 0], xedges) - 1
    binsy = np.digitize(snpathftrs[:, 1], yedges) - 1
    wgts = selection / (h_all[binsx, binsy] * len([sn for sn in snlist if sn.type == typ]) / len(snlist))
    wgts[h_all[binsx, binsy] == 0] = 0
    h_n, _, te, im = ax_hist.hist2d(snpathftrs[:, 0], snpathftrs[:, 1], bins=[xedges, yedges], weights=wgts,
                                    cmap='Blues')
    im.set_norm(Normalize(np.nanpercentile(h_n, botpctl), np.nanmax(h_n)))
    ax_hist.set_xlabel('days since explosion')
    # ylim=ax_hist.get_ylim()
    ax_hist.grid()
    # ax_hist.set_yticks(list(spectral_lines.values()))
    # ax_hist.set_yticklabels(list(spectral_lines.keys()))
    plt.colorbar(im, ax=ax_hist, orientation='horizontal',
                 label='Share among SNe passed through a node /\n Share among SNe in set')
    add_spectral_lines(ax_hist)

    c = plot_meanflux([sn for sn in snlist if sn.type == typ], TIME, LAMB, ax_fl)
    ax_fl.set_xlabel('days since explosion')
    ax_fl.set_ylabel(r'wavelength [$\AA$]')
    plt.colorbar(c, ax=ax_fl, orientation='horizontal', label='Flux rescaled')

    plt.suptitle(typ)
    plt.show()


if __name__ == '__main__':
    snlist, allpaths = examine_rf()

    draw_paths(allpaths, snlist)
    draw_paths(allpaths, snlist, ['Ic-BL'])
    draw_paths(allpaths, snlist, ['Ib', 'Ic'])
    draw_paths(allpaths, snlist, ['Ib'])
    draw_paths(allpaths, snlist, ['Ic'])
    draw_paths(allpaths, snlist, ['II'])
    draw_paths(allpaths, snlist, ['IIn'])
