# Deep-learning-based phase picking for volcano-tectonic and long-period earthquakes
[![DOI](https://zenodo.org/badge/800909138.svg)](https://zenodo.org/doi/10.5281/zenodo.11199021)

This repository contains the final models and the code to reproduce the model (downloading waveforms, formatting data into seisbench format, training and evaluating deep-learning phase pickers).

## Prerequisites
[SeisBench](https://github.com/seisbench/seisbench) is necessary for using our models.

## Model Usage
If you are only interested in applying the model, please see folder `Final_models`, and you can ignore other folders.

~~To load the models in SeisBench, you need to copy them to the model directory of SeisBench.~~  
The model weights have been uploaded to the SeisBench repository and can be accessed through `from_pretrained('volpick')` directly.  `Final_models/demo.ipynb` shows a minimal example. Check the [SeisBench document](https://seisbench.readthedocs.io/en/stable/) for more details about SeisBench model API.

**(1) Models**

```
Final_models
├── demo.ipynb            # a demo
├── volpick               # models presented in the paper, trained on the data splits of 83.6% (training), 5.5% (validation), 10.9% (testing)
└── volpick_95train_5val  # models trained on the data splits of 94.5% (training), 5.5% (validation), 0% (testing), with the above test set included into the training set
```

Here the `volpick` folder contains the models presented in our paper. 

We provide alternative models in folder `volpick_95train_5val`, which were trained by incorporating the test set into the training set but keeping the validation set. Since we had trained and tested `volpick` with a standard train/validation/test data splitting, it might be helpful to make full use of the testing waveforms (including the test examples of LFEs along the Nankai trough and LPs along the Cascadia subduction zone in the paper). We assume the performances of the models in `volpick_95train_5val` are not worse than the models in `volpick`, assuming that a larger dataset tends to give a more robust model. Based on the performances on the same validation set, volpick_95train_5val performs a little bit better than volpick. Note that it might be risky to use the models in the `volpick_95train_5val` because the test set has been "burned". If you are not sure which one to use, we recommend using `volpick`.


**(2) Example snippet**

The following code shows how to load the model in seisbench:
```python
import seisbench.models as sbm
import pandas as pd
...
# First read data into an obspy Stream
...

# Load the picker
picker = sbm.EQTransformer.from_pretrained("volpick")
# or picker = sbm.PhaseNet.from_pretrained("volpick")
print(picker.weights_docstring)

P_threshold=0.2
S_threshold=0.2

# Picking
picks = picker.classify(
          stream, # obspy Stream
          batch_size=256, # batch_size may depend on available memory on your machine
          overlap=5500, # a 55s overlap is recommended for Eqt by Pita‐Sllim et al. 2023 (https://doi.org/10.1785/0320230024)
          blinding=(500, 500),
          stacking="avg",
          parallelism=None, # https://github.com/seisbench/seisbench/issues/272
          P_threshold=P_threshold,
          S_threshold=S_threshold,
          copy=True,
          # copy: If true, copies the input stream. Otherwise, the input stream is modified in place.
          # See https://github.com/seisbench/seisbench/commit/862d9ee708c2c3e737da4e90ab3355471aa01ecf
      ).picks

# save the result
def picklist2df(picks: sbu.PickList):
    def pick2dict(p: sbu.Pick):
        return {
            "trace_id": p.trace_id,
            "start_time": p.start_time,
            "end_time": p.end_time,
            "peak_time": p.peak_time,
            "peak_value": p.peak_value,
            "phase": p.phase,
        }

    pick_df = [pick2dict(p) for p in picks]
    pick_df = pd.DataFrame(pick_df)
    return pick_df
pick_df = picklist2df(picks)
pick_df.to_csv("picks.csv",index=False)
```



# Dataset

For those who are interested in benchmarking their methods on  volcanic seismic waveforms, we provide the `VCSEIS` benchmark dataset, which contains local earthquakes from volcanic regions cataloged by Alaska Volcano Observatory, Hawaiian volcano observatory, Northern California Earthquake Data Center, Pacific Northwest Seismic Network, and compiled into SeisBench format by Zhong and Tan (2024). This dataset is a subset of the dataset in Zhong and Tan (2024), with the data from Japan excluded.

The data set can be loaded as `dataset = sbd.VCSEIS()`. Data from different regions can be selected using the the `get_[region]_subset()` function.

```python
import seisbench.data as sbd

dataset = sbd.VCSEIS()

alaska = dataset.get_alaska_subset()  # select the data from Alaska

hawaii = dataset.get_hawaii_subset()  # select the data from Hawaii

nca = dataset.get_northern_california_subset() # select the data from Northern California

cascade = dataset.get_cascade_subset()  # select the data from Cascade

lp_eq = dataset.get_long_period_earthquakes() # select long-period earthquakes

regular_eq = dataset.get_regular_earthquakes() # select regular/vt earthquakes

noise = dataset.get_noise_traces() # select noise traces
```


## Python scripts
>Note: If you just need to use the final model, you can ignore these scripts. Please see the *Model Usage* section.

The scripts and notebooks represent the workflow we used to process data, train and evaluate deep-learning phase pickers on volcano seismicity in the paper. The scripts are not designed as a package for general purposes, so there are inevitably some quick and dirty code implementations. 

**Summary of some files**

`volpick/data/data.py`: a module for downloading and formatting data

`volpick/data/convert.py` convert downloaded mseed files and original metadata to hdf5+csv files in seisbench
format

`volpick/data/utils.py`: functions for calculating signal to noise ratios, frequency indices

`volpick/model/model.py`: a module defining training workflows in the convention of pytorch lightning.

`volpick/model/train.py`: provides an interface for training models.

`volpick/model/eval*.py`: funtions for model evaluations



`model_training`: notebooks and scripts that invoke scripts from the volpick folder to train and evaluate models.

## Reference
Zhong, Y., & Tan, Y. J. (2024). Deep‐learning‐based phase picking for volcano‐tectonic and long‐period earthquakes. Geophysical Research Letters, 51, e2024GL108438. https://doi.org/10.1029/2024GL108438

## Projects using volpick
- Wei, J., Liu, Q., Chen, L., Wei, S., & Zhao, L. (2024). A novel 3-D seismic scattering and intrinsic attenuation tomography and its application to Northern Sumatra. Journal of Geophysical Research: Solid Earth, 129, e2024JB029116. https://doi.org/10.1029/2024JB029116
- Fountoulakis, I., & Evangelidis, C. P. (2025). The 2024–2025 seismic sequence in the Santorini-Amorgos region: Insights into volcano-tectonic activity through high-resolution seismic monitoring. Seismica, 4(1).

## Acknowledgement
Part of the training and evaluation code is adapted from [pick-benchmark](https://github.com/seisbench/pick-benchmark).

**We are looking for more LP data with analyst picks to improve our model. We would greatly appreciate it if you have the data and are willing to contribute.**
