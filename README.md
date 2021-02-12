Exploring the diversity of SNe, by their flux function through time and wavelength. Meant to be used on output from [PyCoCo](https://github.com/ofek-b/PyCoCo_templates).

This code is used in a work in progress.

### Prerequisites

- Tested on Python 3.8, requires the basic packages
- [empca.py](https://github.com/sbailey/empca/blob/master/empca.py) in an importable directory
- [scikit-learn](https://scikit-learn.org/stable/)
- [spectres](https://github.com/ACCarnall/spectres)

### Files:

`reduction.py`: Routines for dimensionality reduction and other forms of data analysis, to be used on the data.

`snfuncs.py`: Some input + SN object class and functions for managing the data.

`utils.py`: Some input + directories, visialization etc.

`main.py`: Running everything.

`pickled_snlist`: Pickled list of SN objects, to spare recalculation of features, for when it takes long.