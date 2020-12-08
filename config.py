from os import environ, system
from os.path import join
import numpy as np
import matplotlib as mpl
import pandas as pd

"""directories:"""
SNLISTS_DIR = join(environ['HOME'], 'DropboxWIS/spectra_in_time/pickled')
PYCOCO_DIR = join(environ['HOME'], 'DropboxWIS/PyCoCo_templates')

"""visuals:"""
typ2color = {
    'Ia': 'black',
    'Ib': '#1f77b4',  # blue
    'Ic': '#2ca02c',  # green
    'Ic-BL': 'lime',
    'II': '#d62728',  # red
    'IIn': '#ff7f0e',  # orange
    'IIb': '#9467bd',  # purple
    '': '#8c564b',  # brown
    '': '#e377c2',  # pink
    '': '#7f7f7f',  # grey
    '': '#bcbd22',  # yellow-green
    '': '#17becf'  # turquoise
}

marker = 'o'
size = 80

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['legend.title_fontsize'] = 10
# mpl.rcParams['legend.fancybox'] = True

lambstr = r'wavelength [$\AA$]'
fluxstr = r'$f_\lambda$ [erg s-1 cm-2 $\AA$-1]'
timestr = r'rest days - max $m_B$'

"""etc:"""
NUM_JOBS = 16

"""Not input:"""

# with host correction:
# PYCOCO_SED_PATH = join(PYCOCO_DIR, 'Templates_HostCorrected/pycoco_%s.SED')
# PYCOCO_FINAL_DIR = join(PYCOCO_DIR, 'Outputs/%s/FINAL_spectra_2dim')

# without host correction:
PYCOCO_SED_PATH = join(PYCOCO_DIR, 'Templates_noHostCorr/pycoco_%s_noHostCorr.SED')
PYCOCO_FINAL_DIR = join(PYCOCO_DIR, 'Outputs/%s/FINAL_spectra_2dim/HostNotCorr')

PYCOCO_INFO_PATH = join(PYCOCO_DIR, 'Inputs/SNe_Info/info.dat')

system('mkdir -p ' + SNLISTS_DIR)

info_df = pd.read_csv(PYCOCO_INFO_PATH, delimiter=' ', index_col=0)
# info_df.set_index('Name')
# nm2typ = {nm: typ for nm, typ in zip(info_df['Name'], info_df['Type'])}


"""Results-critical:"""

timeclipdict = {'SN2006aj': (2.5, np.inf)}  # from the original time count, this in the pycoco output

band_wvl, band_throughput = np.genfromtxt(join(PYCOCO_DIR, 'Inputs/Filters/GeneralFilters/', 'Bessell_B.dat'),
                                          unpack=True)  # for maxtime calculation

timeclip_sincemax = (-5, 20)  # time count here is time since max
LAMB = np.arange(4000, 8000, 20)  # AA

if __name__ == '__main__':
    print(info_df['Type'].to_list())
