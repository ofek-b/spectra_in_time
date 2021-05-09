from analysis import *


def twospec_vs_displacement(snname, fixed_mjd):
    # train normally, remember distances of original SN to compare later
    exc, snlist_, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train(additional_exc=[snname])
    friends = [i for i, sn in enumerate(snlist_) if sn.type == 'Ia']  # todo: input
    dismat_cluster = build_dissimilarity_matrix(X_PC[friends, :])
    cluster_diam = np.max(dismat_cluster)

    # dismat = build_dissimilarity_matrix(X_PC)
    # idx = next(i for i,sn in enumerate(snlist_) if sn.name==snname)
    # args = [a for a in np.argsort(dismat[idx,:]) if a!=idx]
    # friendidx=args[0]
    # friendname = snlist_[args[0]].name
    # dist2friend=dismat[idx,friendidx]
    # args = args[:5]
    # closest = np.array([dismat[idx,a] for a in args])

    # exc, snlist_, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train(additional_exc=['SN2005cf'])
    # friendidx = next(i for i, sn in enumerate(snlist_) if sn.name == friendname)

    original_sn = SN(snname)
    degraded_sne = [SN(nm) for nm in info_df.index.to_list() if nm.startswith(snname + '_' + str(fixed_mjd))]
    sne_x_PC = []
    for sn in [original_sn] + degraded_sne:
        # print(sn.exptime)
        query_x = calcfeatures([sn])
        query_x[np.isnan(query_x)] = 0
        query_weights = ~np.isnan(query_x) + 0
        # apply trained rf on query SN:
        query_x = scaler.transform(query_x.reshape(1, -1))
        m.set_data(query_x, query_weights)
        query_x_PC = m.model
        sne_x_PC.append(query_x_PC)

    sn0 = degraded_sne[0]
    time_fixed_spec = int(sn0.name.split('_')[1]) - sn0.exptime
    times_2nd_specs = [int(sn.name.split('_')[2]) - sn.exptime for sn in degraded_sne]

    dismat_q = build_dissimilarity_matrix(np.row_stack([sne_x_PC[0], X_PC]))
    # friends = np.array([a for a in np.argsort(dismat_q[0, :]) if a != 0]) - 1
    # friends = friends[:5]
    friends = np.array(friends)
    dist2friends = dismat_q[0, friends + 1]
    orig_mean_to_diam = np.mean(dist2friends) / cluster_diam

    dismat_q = build_dissimilarity_matrix(np.row_stack(sne_x_PC[1:] + [X_PC]))
    dist2friends_for_degraded = dismat_q[:len(sne_x_PC) - 1, :][:, len(sne_x_PC) - 1 + friends]
    comparison = np.mean(dist2friends_for_degraded, axis=1) / cluster_diam
    # s=np.mean(np.abs(comparison - closest) / closest,axis=1)

    maxtime = timesincemax(original_sn.time, original_sn.lamb, original_sn.flux)
    plt.scatter(times_2nd_specs, comparison)
    ax = plt.gca()
    plt.axvline(maxtime, linestyle='--', color='grey')
    ax.annotate('B-max', xy=(maxtime, 0.85), xycoords=('data', 'axes fraction'))
    plt.axhline(orig_mean_to_diam, linestyle='--', color='grey')
    ax.annotate('full data', xy=(0.85, orig_mean_to_diam), xycoords=('axes fraction', 'data'))
    ax.annotate('%s\nfixed 1st spec @ %.0f d' % (snname, time_fixed_spec), xy=(0.85, 0.85), xycoords='axes fraction')

    # plt.title('dissim(%s,%s with spectra at %.0f days since exp. and at t). Originally %s'%(friendname,snname,time_fixed_spec,dismat_q[0,1]))
    # plt.title('Effect of 2nd spectrum`s epoch, 1st spectrum at %.0f days after exp.'%time_fixed_spec)
    plt.ylabel('mean dissim. to cluster / cluster diam')
    plt.xlabel('epoch of 2nd spectrum [days since exp.]')
    plt.grid()
    plt.show()


def onespec_vs_displacement(snname, spectimes, timewindow=4):
    exc, snlist_, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()
    dismat = build_dissimilarity_matrix(X_PC)
    idx = next(i for i, sn in enumerate(snlist_) if sn.name == snname)
    args = [a for a in np.argsort(dismat[idx, :]) if a != idx]
    args = args[:5]
    closest = np.array([dismat[idx, a] for a in args])

    # degraded_sn = []
    self_dists, spectimes_real = [], []
    for spectime in spectimes:
        sn = SN(snname)
        specidx_real = np.argmin(abs(sn.time - spectime))
        spectime_real = sn.time[specidx_real]
        spectimes_real.append(spectime_real)
        keep = np.abs(sn.time - spectime_real) <= timewindow / 2
        sn.time, sn.flux = sn.time[keep], sn.flux[keep, :]

        query_x = calcfeatures([sn])
        query_x[np.isnan(query_x)] = 0
        query_weights = ~np.isnan(query_x) + 0
        # apply trained rf on query SN:
        query_x = scaler.transform(query_x.reshape(1, -1))
        m.set_data(query_x, query_weights)
        query_x_PC = m.model
        dismat_q = build_dissimilarity_matrix(np.row_stack([X_PC[args, :], query_x_PC]))
        comparison = dismat_q[-1, :-1]
        s = np.mean(np.abs(comparison - closest) / closest)
        self_dists.append(s)

    plt.plot(spectimes_real, self_dists)
    plt.show()

    # matrix = np.column_stack((orig_dists, new_dists))
    # snlist = [sn_ for sn_ in snlist_ if sn_.name != snname]
    # names = [sn.name for sn in snlist]
    # fulltypes = [sn.type for sn in snlist]
    # colors = [typ2color[typ] for typ in fulltypes]
    # view_dim = 2
    # ax = plt.axes(projection='3d' if view_dim == 3 else None)
    # ax.plot([min(orig_dists), max(orig_dists)],[min(orig_dists), max(orig_dists)],alpha=0.5,linestyle='--',color='k')
    # pctgs = np.ones((len(colors), 1))
    # pctg2sz = lambda x: size * x
    # scat = ax.scatter(matrix[:, 0], matrix[:, 1], c=colors, marker=marker, s=pctg2sz(pctgs), alpha=alpha)
    # ax.set_xlabel('Dissimilarity with '+snname)
    # ax.set_ylabel( 'Dissimilarity with degraded data')
    # annotate_onclick(scat, matrix, names)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    pass

    twospec_vs_displacement('SN2005cf', 53533)

    # onespec_vs_displacement('SN2016coi',[3,5,8,10,15,20,25,30,40])
