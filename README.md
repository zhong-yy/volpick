# Deep-learning-based phase picking for volcano-tectonic and long-period earthquakes
[![DOI](https://zenodo.org/badge/800909138.svg)](https://zenodo.org/doi/10.5281/zenodo.11199021)

This repository contains code for downloading waveforms, formatting data into seisbench format, training deep-learning phase pickers on volcanic earthquake waveforms and evaluating model performances.



# Usage
For model users, only the `Final_models` folder is important. 

[SeisBench](https://github.com/seisbench/seisbench) is required to load and run the models.

See `Final_models/demo.ipynb`.

The following code shows how to load the model in seisbench:
```python
import seisbench.models as sbm
picker = sbm.PhaseNet.from_pretrained("volpick")
print(picker.weights_docstring)
```




# Code
The code here is for downloading and formatting data, training and testing a model.

