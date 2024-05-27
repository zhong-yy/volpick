# Deep-learning-based phase picking for volcano-tectonic and long-period earthquakes
[![DOI](https://zenodo.org/badge/800909138.svg)](https://zenodo.org/doi/10.5281/zenodo.11199021)

This repository contains code for downloading waveforms, formatting data into seisbench format, training deep-learning phase pickers on volcanic earthquake waveforms and evaluating model performances.



# Usage
For model users, only the `Final_models` folder is important. The other files are not required.

[SeisBench](https://github.com/seisbench/seisbench) is required to load and run the models.

See `Final_models/demo.ipynb`.

The following code shows how to load the model in seisbench:
```python
import seisbench.models as sbm
picker = sbm.PhaseNet.from_pretrained("volpick")
print(picker.weights_docstring)
```




# Python scripts
`volpick/data/data.py`: a module for downloading and formatting data

`volpick/data/convert.py` convert downloaded mseed files and original metadata to hdf5+csv files in seisbench
format

`volpick/data/utils.py`: functions for calculating signal to noise ratios, frequency indices

`volpick/model/model.py`: a module defining training workflows in the convention of pytorch lightning.

`volpick/model/train.py`: provides an interface for training models.

`volpick/model/eval*.py`: funtions for model evaluations



`model_training`: notebooks and scripts that invoke scripts from the volpick folder to train and evaluate models.



These scripts and notebooks are not designed as a package for general purposes. They just represent the workflow we used to train and evaluate deep-learning phase picker on volcano seismicity in the paper. Therefore, there are inevitably some quick and dirty code implementations that may not be used directly in other situations.

Part of the training and evaluation code is adapted from [pick-benchmark](https://github.com/seisbench/pick-benchmark).

