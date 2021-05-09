from analysis import *


def stripe_index(dismat, idxs):
    dismat_ = dismat[idxs,:][:,idxs]
    np.fill_diagonal(dismat_,np.nan)
    s = np.nanmax(dismat_)/np.nanmean( dismat_ )
    return s

def manual_clustering():
    exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()  # train using the template SNe
    dismat = build_dissimilarity_matrix(X_PC)


    clusters = dict(
        main_II=[sn.name for sn in snlist if sn.type == 'II'],
        main_IIn=[sn.name for sn in snlist if sn.type == 'IIn'],
        main_Ia=[sn.name for sn in snlist if sn.type == 'Ia'],
        main_IIb=['SN2011hs', 'SN2011dh', 'SN2006T', 'SN2008bo', 'SN2013df', 'SN2011fu', 'SN1993J'],
        main_Ib=['SN2004gv', 'SN2007Y', 'SN2006ep', 'SN2015ah', 'SN1999dn', 'iPTF13bvn', 'SN2005hg', 'SN2009iz'],
        main_Ic=['SN2013ge', 'SN1994I', 'SN2009jf', 'SN2007gr', 'SN2014L', 'SN2004fe'],
        main_IcBL=['SN2006aj', 'SN2019gwc', 'SN2003jd', 'SN2012ap', 'SN2009bb', 'SN1998bw', 'SN2007ru'],
        rest_Ib=['SN2004gq', 'SN2015ap', 'SN2007uy', 'SN2016jdw', 'SN2017bgu', 'SN2004dk'], )

    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_rest_Ib, label='Cluster "Rest Ib"', comp_name='SN2016jdw')
    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_rest_Ib, label='Cluster "Rest Ib"', comp_name='iPTF13bvn')

    showmean(snlist, X_PC, TIME, LAMB, dlog=True, names=clusters['main_Ib'], label='Main Cluster Ib',
             comp_name='SN2004gq',
             speclines=False)
    showmean(snlist, X_PC, TIME, LAMB, dlog=True, names=clusters['rest_Ib'], label='Rest Ib', comp_name='iPTF13bvn',
             speclines=False)
    # showmean(snlist, X_PC, TIME, LAMB, names=clusters['main_Ib'], label='Cluster "Main Ib"', comp_name='SN2008ax',
    #          speclines=False)

    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IIb, label='Cluster "Main IIb"', comp_name='SN2008ax')
    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IIb, label='Cluster "Main IIb"', comp_name='SN2008bo')
    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IIb, label='Cluster "Main IIb"', comp_name='SN2011dh')

    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IcBL, label='Cluster "Main Ic-BL"', comp_name='SN2002ap')
    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_Ic, label='Cluster "Main Ic"', comp_name='SN2002ap')
    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IcBL, label='Cluster "Main Ic-BL"', comp_name='ZTF20abwxywy',speclines=False)
    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_Ic, label='Cluster "Main Ic"', comp_name='ZTF20abwxywy',speclines=False)
    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IcBL, label='Cluster "Main Ic-BL"', comp_name='SN2014L')
    # showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_Ic, label='Cluster "Main Ic"', comp_name='SN2014L')

    # showmean(snlist,X_PC, TIME, LAMB,cluster_rest_Ib, comp_name='iPTF13bvn')


def rigorous_clustering():
    exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()  # train using the template SNe


def query(query_names):
    exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()  # train using the template SNe

    dissims_to_training, query_snlist = [], []
    for nm in query_names:
        query_sn = SN(nm) if isinstance(nm,str) else nm
        query_sn.type = query_sn.name + ' (query)'
        query_x = calcfeatures([query_sn])
        query_x[np.isnan(query_x)] = 0
        query_weights = ~np.isnan(query_x) + 0

        # apply trained rf on query SN:
        query_x = scaler.transform(query_x.reshape(1, -1))
        m.set_data(query_x, query_weights)
        query_x_PC = m.model
        dismat_q = build_dissimilarity_matrix(np.row_stack([X_PC, query_x_PC]))
        dissims_to_training.append(dismat_q[-1, :-1])

        dismat_emb = tsne_(dismat_q, n_components=3, perplexity=10, learning_rate=15)
        myscatter(dismat_emb, snlist+[query_sn], save_anim=False)  # 2D or 3D scatter plot



        query_snlist.append(query_sn)

    return dissims_to_training, query_snlist, snlist, build_dissimilarity_matrix(X_PC)


if __name__ == '__main__':
    # dissims_to_training, query_snlist, snlist, dismat = query([
    # 'SN1987A', 'SN2009ip', 'SN2010al', 'SN2005bf', 'SN2013gh', 'SN2011bm', 'SN2016bkv', 'SN2017hyh', 'SN2012cg',
    #    'SN2008D','SN2005cp'
    #     'SN2005cp'
    # ])
    # dismatplot_query(dissims_to_training, query_snlist, snlist, dismat)  # view dissimilarities with training set
    manual_clustering()
    pass
    # query(['slimSN2011bm'])
