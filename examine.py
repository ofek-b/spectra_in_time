import itertools

from matplotlib.colors import LogNorm
from tqdm import tqdm

from analysis import *

rng = [[TIME[0], TIME[-1] + TIME[1] - TIME[0]], [LAMB[0], LAMB[-1] + LAMB[1] - LAMB[0]]]
xedges, yedges = np.linspace(rng[0][0], rng[0][1], 10), np.linspace(rng[1][0], rng[1][1], 50)


def examine_rf(indices=False):
    exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()
    indicator, n_nodes_ptr = rand_f.decision_path(X_PC)
    filler = np.zeros((0, 1 if indices else 2))

    def eatimator_paths(i):
        clf = rand_f.estimators_[i]
        sn_treepaths = [filler for _ in snlist]  # if not a good tree for this SN, appned an empty path
        for snidx in np.where(clf.predict_proba(X_PC)[:, 0] == 1)[0]:
            ftr_indices = clf.tree_.feature[
                (indicator.getrow(snidx).toarray().flatten()[n_nodes_ptr[i]:n_nodes_ptr[i + 1]] != 0)
                & (clf.tree_.feature >= 0)]
            if indices:
                sn_treepaths[snidx] = np.array(ftr_indices)
            else:
                tind, lind = np.unravel_index(ftr_indices, (len(TIME), len(LAMB)))
                ftrs = np.column_stack((TIME[tind], LAMB[lind]))
                sn_treepaths[snidx] = ftrs

        return sn_treepaths

    allpaths = [eatimator_paths(i) for i in tqdm(range(len(rand_f.estimators_)))]
    allpaths = list(zip(*allpaths))  # allpaths[sn_idx][i] = path of sn_idx through good tree i

    return snlist, allpaths, rand_f, X_PC


def change_features_distance():
    snlist, allpaths, rand_f, X_PC = examine_rf(indices=True)
    n_sn, n_f = X_PC.shape

    def get_feature_importance(sn_idx):
        line = X_PC[sn_idx, :]
        others = X_PC[[i for i in range(n_sn) if i != sn_idx], :]
        good_ests_idx = [i for i, est in enumerate(rand_f.estimators_)
                         if est.predict_proba(line.reshape(1, -1))[0, 0] == 1]
        good_ests = [rand_f.estimators_[i] for i in good_ests_idx]
        line_output = np.column_stack([est.apply(line.reshape(1, -1)) for est in good_ests]).flatten()

        features_per_goodest = [list(set(allpaths[sn_idx][i])) for i in good_ests_idx]

        def e(i):
            lst = [list(range((n_sn - 1) * ft, (n_sn - 1) * (ft + 1))) for j, ft in enumerate(features_per_goodest[i])]
            a = []
            for l in lst:
                a.extend(l)
            lst = [[ft] * (n_sn - 1) for j, ft in enumerate(features_per_goodest[i])]
            b = []
            for l in lst:
                b.extend(l)
            modifications = np.tile(line, ((n_sn - 1) * len(features_per_goodest[i]), 1))
            modifications[range(modifications.shape[0]), b] = others[:, features_per_goodest[i]].T.flatten()

            tree_good_for = good_ests[i].predict_proba(modifications)[:, 0] == 1
            apply_mat = np.zeros(modifications.shape[0]) + np.nan
            apply_mat[tree_good_for] = good_ests[i].apply(modifications[tree_good_for, :])
            return a, apply_mat

        d = np.zeros(((n_sn - 1) * n_f, len(good_ests)), dtype=np.bool)
        for i in range(len(good_ests)):
            a, apply_mat = e(i)
            d[a, i] = apply_mat != line_output[i]
        d = np.nanmean(d, axis=1)  # over all trees, w/o those not good for the modification
        d = d.reshape((n_f, n_sn - 1)).T
        # d is the distance of original from itself with a single feature replaced with its value in another SN
        return d

    ds = Parallel(n_jobs=8, verbose=11)(delayed(get_feature_importance)(sn_idx) for sn_idx in range(n_sn))
    # ds= [get_feature_importance(sn_idx) for sn_idx in tqdm(range(n_sn))]
    ds = np.stack(ds, axis=0)
    return ds, snlist


def sntest_vs_snwanted_importance(num_features_2modify, sntest_idx, snwanted_idx, X_PC, build_dissimilarity_matrix,
                                  rand_f):
    n_sn, n_f = X_PC.shape

    line_t, line_w = X_PC[sntest_idx, :], X_PC[snwanted_idx, :]
    orig_dist = build_dissimilarity_matrix(np.row_stack([line_t, line_w]), info=False)
    orig_dist = orig_dist[0, 1]
    good_trees = [np.all(est.predict_proba(np.row_stack((line_t, line_w)))[:, 0] == 1) for est in rand_f.estimators_]

    apply_line_w = rand_f.apply(line_w.reshape((1, -1)))
    apply_line_w = apply_line_w[0, good_trees]

    fpairs = list(itertools.combinations(range(n_f), num_features_2modify))

    n_p = len(fpairs)
    step = int(2e4)
    batches = np.arange(0, n_p, step)

    dist_by_feature_modified = np.array([])
    for b in tqdm(batches):
        fpairs_batch = fpairs[b:min([b + step, n_p])]
        mods = np.tile(line_t, (len(fpairs_batch), 1))
        for ordinal in range(num_features_2modify):
            i_ = list(list(zip(*fpairs_batch))[ordinal])
            mods[range(len(fpairs_batch)), i_] = line_w[i_]
        apply_mat = rand_f.apply(mods)
        apply_mat = apply_mat[:, good_trees]
        dist_by_feature_modified_batch = np.mean(apply_mat != apply_line_w, axis=1)
        dist_by_feature_modified = np.concatenate((dist_by_feature_modified, dist_by_feature_modified_batch))
    return fpairs, 1 - dist_by_feature_modified / orig_dist


def draw_paths(allpaths, snlist, typs=None, num=4):
    if typs is not None:
        idxs = [i for i, sn in enumerate(snlist) if sn.type in typs]
        allpaths = [allpaths[i] for i in idxs]
        snlist = [snlist[i] for i in idxs]
    num_goodpaths = 0
    g = {}  # a fully connected graph with nodes as features (binned).
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
    for e in g:
        x, y = zip(e[0], e[1])
        x, y = np.array(x), np.array(y)
        x = (xedges[x] + xedges[x + 1]) / 2
        y = (yedges[y] + yedges[y + 1]) / 2
        ax_fl.plot(x, y, color=plt.cm.summer((g[e] - minweight) / (maxweight - minweight)), alpha=1, marker='*')
    norm = mpl.colors.Normalize(vmin=minweight, vmax=maxweight)
    sm = plt.cm.ScalarMappable(cmap='summer', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax_c1, orientation='horizontal', pad=0.2,
                 label='% of valid SN paths with\n a feature from each 2d bin')

    c = plot_meanflux(snlist, TIME, LAMB, ax_fl)
    ax_fl.set_xlabel('days since explosion')
    ax_fl.set_ylabel(r'wavelength [$\AA$]')
    plt.colorbar(c, cax=ax_c2, orientation='horizontal', pad=0.2,
                 label=r'$\tilde{f}(t, \lambda)$')
    add_spectral_lines(ax_fl, cl='k', allalong=True)

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
    # fl = join(environ['HOME'], 'DropboxWIS/spectra_in_time/importance.pickle')
    # if isfile(fl):
    #     with open(fl, 'rb') as f:
    #         ds, snlist = pickle.load(f)
    # else:
    #     ds, snlist = change_features_distance()
    #     with open(fl, 'wb') as f:
    #         pickle.dump((ds, snlist), f)

    # sn_idx = [ii for ii, sn in enumerate(snlist) if sn.type == 'Ia']
    # rep_sn_idx = [ii for ii, sn in enumerate(snlist) if sn.type == 'IIn']
    # m = ds[sn_idx,:,:]
    # m = np.mean(m,axis=(0 ,1))

    # params=[]
    # for t in sn_test_idx:
    #     for w in sn_wanted_idx:
    #         params.append((t,w))

    # tuples = Parallel(n_jobs=1,verbose=11)(
    #     delayed( sntest_vs_snwanted_importance)(2,*p, X_PC,build_dissimilarity_matrix, rand_f ) for p in params)

    fl = join(environ['HOME'], 'DropboxWIS/spectra_in_time/importance_1.pickle')
    if isfile(fl):
        with open(fl, 'rb') as f:
            edges, importance = pickle.load(f)
    else:
        exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()
        exc, snlist = train(onlymeta=True)
        sn_test_idx = [ii for ii, sn in enumerate(snlist) if sn.name == 'SN2009jf'][0]
        sn_wanted_idx = [ii for ii, sn in enumerate(snlist) if sn.name == 'iPTF13bvn'][0]
        edges, importance = sntest_vs_snwanted_importance(
            1, sn_test_idx, sn_wanted_idx, X_PC, build_dissimilarity_matrix, rand_f)
        with open(fl, 'wb') as f:
            pickle.dump((edges, importance), f)

    # mn,mx = min(importance), max(importance)
    # sorted_idxs = sorted(range(len(importance)), key=lambda ii: importance[ii],reverse=True)
    # for k in sorted_idxs[:5]:
    #     tind, lind = np.unravel_index(edges[k], (len(TIME), len(LAMB)))
    #     plt.plot(TIME[tind], LAMB[lind], color=plt.cm.summer((importance[k] - mn) / (mx - mn)), alpha=1, marker='*')
    #
    # norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
    # sm = plt.cm.ScalarMappable(cmap='summer', norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, orientation='horizontal', pad=0.2,
    #              label='% of valid SN paths with\n a feature from each 2d bin')
    #
    # plt.show()

    # m=np.mean(np.row_stack(ms),axis=0)
    # m=100*m

    """plot:"""

    # m = np.zeros(5151)
    # for l in zip(*edges):
    #     m[list(l)]+=importance

    exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()
    m = rand_f.feature_importances_

    # norm = Normalize(np.min(m), np.max(m))
    # norm = Normalize(np.percentile(m, 90), np.max(m))

    ax1, ax2 = plt.axes([0.1, 0.1, 0.3, 0.8]), plt.axes([0.45, 0.1, 0.5, 0.8])
    # ax2=plt.gca()

    m = m.reshape(len(TIME), len(LAMB))
    im = ax2.imshow(m[:, ::-1].T, cmap='Blues', aspect='auto', interpolation='none', norm=None,
                    extent=[TIME[0], TIME[-1], LAMB[0], LAMB[-1]])
    add_spectral_lines(cl='pink', allalong=True, ax=ax2)
    # ax2.set_title(snlist[idx].name)
    plt.colorbar(mappable=im, ax=ax2, pad=0.15, shrink=0.3, label='Feature Importance')
    ax2.set_ylabel(r'wavelength [$\AA$]')
    ax2.set_xlabel('days since explosion')

    wgts = np.sum(m, axis=0)
    ax1.hist(LAMB, weights=wgts, bins=len(LAMB) // 4, align='mid', orientation='horizontal')
    ax1.set_ylim(LAMB[0], LAMB[-1])
    ax1.invert_xaxis()
    ax1.set_ylabel(r'wavelength [$\AA$]')
    # ax1.set_xlabel('Sum at wavelength bin - mean over bins')

    plt.show()

    # snlist, allpaths = examine_rf()

    # draw_paths(allpaths, snlist)
    # draw_paths(allpaths, snlist, ['Ic-BL'])
    # draw_paths(allpaths, snlist, ['Ib', 'Ic'])
    # draw_paths(allpaths, snlist, ['Ib'])
    # draw_paths(allpaths, snlist, ['Ic'])
    # draw_paths(allpaths, snlist, ['II'])
    # draw_paths(allpaths, snlist, ['IIn'])
