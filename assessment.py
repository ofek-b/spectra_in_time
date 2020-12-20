import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from utils import info_df


def sill(snlist, reduced_data):
    fulltypes = [info_df['FullType'][sn.name] if not pd.isna(info_df['FullType'][sn.name]) else info_df['Type'][sn.name]
                 for sn in snlist]

    sils = silhouette_samples(reduced_data, fulltypes, metric='euclidean')

    for typ in set(fulltypes):
        plt.hist(sils[np.array(fulltypes) == typ])
        plt.title(typ)
        plt.show()
