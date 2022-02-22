Exploring the diversity of SNe, by their flux function through time and wavelength. Meant to be used on output from [PyCoCo](https://github.com/ofek-b/PyCoCo_templates).

This code is used in [Bengyat & Gal-Yam (2022)](https://arxiv.org/abs/2202.10300).

### Prerequisites

- Tested on Python 3.8, requires the basic packages
- Clone [PyCoCo](https://github.com/ofek-b/PyCoCo_templates) and run it on the SNe you wish to use. Or, to run
  spectra_in_time on the SNe used in the paper, use the existing `info.dat` and receive the `/Outputs` folder
  upon [request](mailto:ofek.bengiat@weizmann.ac.il). Alternatively, you can use the 67 templates from
  the [original PyCoCo repo](https://github.com/maria-vincenzi/PyCoCo_templates).
- [empca.py](https://github.com/sbailey/empca/blob/master/empca.py) in an importable directory
- [scikit-learn](https://scikit-learn.org/stable/)
- [spectres](https://github.com/ACCarnall/spectres)

### Files:

`main.py`: Running everything.

`analysis.py`: The Random Forest and data analysis routines.

`snfuncs.py`: Some input + SN object class and functions for managing the data.

`utils.py`: Some input + directories, visialization etc.

`snlist.pickle`: Pickled list of SN objects, to spare recalculation of features, for when it takes long.

post-analysis shown in the paper:

`data_vs_displacement.py`: testing use on degraded data

`clusters.py`: examining average spectra of clusters
