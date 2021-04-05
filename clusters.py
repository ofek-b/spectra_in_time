from analysis import *


def manual_clustering():
    exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()  # train using the template SNe

    cluster_main_Ib = ['SN2007Y', 'SN2006ep', 'SN2015ah', 'SN1999dn', 'iPTF13bvn', 'SN2005hg', 'SN2009iz']
    cluster_main_IIb = ['SN2011hs','SN2011dh','SN2006T','SN2008bo'] # 'SN2008aq','SN2017gpn'
    cluster_main_IcBL = ['SN2006aj', 'ZTF19aaxfcpq', 'SN2003jd', 'SN2012ap', 'SN2009bb', 'SN1998bw']
    cluster_main_Ic = ['SN2004aw', 'SN2013ge', 'SN1994I', 'SN2009jf', 'SN2007gr']

    cluster_rest_Ib = ['SN2004gq', 'SN2015ap', 'SN2004gv', 'SN2007uy', 'SN2016jdw', 'SN2017bgu']

    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IIb, label='Cluster "Main IIb"', comp_name='SN2008ax')
    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IIb, label='Cluster "Main IIb"', comp_name='SN2008bo')
    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IIb, label='Cluster "Main IIb"', comp_name='SN2011dh')

    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IcBL, label='Cluster "Main Ic-BL"', comp_name='SN2002ap')
    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_Ic, label='Cluster "Main Ic"', comp_name='SN2002ap')
    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IcBL, label='Cluster "Main Ic-BL"', comp_name='ZTF20abwxywy')
    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_Ic, label='Cluster "Main Ic"', comp_name='ZTF20abwxywy')
    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_IcBL, label='Cluster "Main Ic-BL"', comp_name='SN2014L')
    showmean(snlist, X_PC, TIME, LAMB, names=cluster_main_Ic, label='Cluster "Main Ic"', comp_name='SN2014L')

    # showmean(snlist,X_PC, TIME, LAMB,cluster_rest_Ib, comp_name='iPTF13bvn')


def rigorous_clustering():
    exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()  # train using the template SNe


def query(query_names):
    exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()  # train using the template SNe

    dissims_to_training, query_snlist = [], []
    for nm in query_names:
        query_sn = SN(nm)
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

        query_snlist.append(query_sn)

    return dissims_to_training, query_snlist, snlist, build_dissimilarity_matrix(X_PC)


if __name__ == '__main__':
    #     dissims_to_training, query_snlist, snlist, dismat = query([
    #         'SN2005bf', 'SN2011bm', 'SN2016bkv', 'SN2017hyh', 'SN2012cg', 'SN2012au', 'SN2004gt', 'SN2007ru', 'SN2008D'
    #     ])
    #     dismatplot_query(dissims_to_training, query_snlist, snlist, dismat)  # view dissimilarities with training set
    pass

