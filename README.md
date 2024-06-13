# Deep-learning-based phase picking for volcano-tectonic and long-period earthquakes
[![DOI](https://zenodo.org/badge/800909138.svg)](https://zenodo.org/doi/10.5281/zenodo.11199021)

This repository contains the final models and the code to reproduce the model (downloading waveforms, formatting data into seisbench format, training and evaluating deep-learning phase pickers).



# 1 Model Usage
If you only care about how to use the model, please see folder `Final_models`, and you can ignore the python script files in other folders. See `Final_models/demo.ipynb` for a simple example.

**(1) Prerequisites**

- [SeisBench](https://github.com/seisbench/seisbench) is required to load and run the models.

**(2) Models**
```
Final_models
├── demo.ipynb            # a demo
├── volpick               # models presented in the paper, trained on the data splits of 83.6% (training), 5.5% (validation), 10.9% (testing)
└── volpick_95train_5val  # models trained on the data splits of 94.5% (training), 5.5% (validation), 0% (testing), with the above test set included into the training set
```

Here the models in the `volpick` folder is the models presented in our paper. We also provide alternative models in folder `volpick_95train_5val`. They were trained by incorporating the test set into the training set. Since we had tested the models, it might be helpful to use the testing waveforms (including those along the Nankai trough and the Cascadia subduction zone) for training. It is risky to use the models in `volpick_95train_5val` because the test set has been "burned". However, we expect that the performances of the models in `volpick_95train_5val` are not worse than the models in `volpick`, based on the assumption that a larger dataset tends to give a more robust model. If you are not sure which one to use, we recommend using `volpick`.

~~To load the models in SeisBench, you need to copy them to the model directory of SeisBench.~~  The model weights have been uploaded to the SeisBench repository and can be used through `from_pretrained('volpick')` or `from_pretrained('volpick_95train')` directly.

**(3) Example snippet**

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






# 2 Python scripts
>Note: If you just want to use the final model, you don't have to look at these python scripts. Please see the *Model Usage* section.

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

# Reference
Zhong, Y., & Tan, Y. J. (2024). Deep‐learning‐based phase picking for volcano‐tectonic and long‐period earthquakes. Geophysical Research Letters, 51, e2024GL108438. https://doi.org/10.1029/2024GL108438

# Acknowledgement
Part of the training and evaluation code is adapted from [pick-benchmark](https://github.com/seisbench/pick-benchmark).

