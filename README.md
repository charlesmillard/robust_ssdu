# Noisier2Noise for reconstruction
**A framework for self-supervised MR image reconstruction and denoising**

_Charles Millard, Mark Chiew_


This repository reproduces results from our self-supervised reconstruction [paper](https://ieeexplore.ieee.org/abstract/document/10194985) 
and self-supervised reconstruction and denoising [paper](https://arxiv.org/abs/2210.01696).

[BibTeX](#citation)


![A schematic of the proposed self-supervised reconstruction and denoising method](./flowchart_denoising.svg)

## Usage 


### Training
To train a network, run

```bash
python train_network.py -c config_name -l log_loc
```
where `config_name` is the  name of one of the configuration files in the configs folder
and `log_loc` is the root for saving the model and tensorboard summary. 

For instance,
```bash
python train_network.py -c default.yaml -l saved/logs/cpu/default/ 
```
trains according to the configuration in the `default.yaml` file and saves the result in the 
directory `saved/logs/cpu/default/`. 

We have provided an example configuration file for each of the training methods in the paper. All of the example configurations
are for 8x column-wise sub-sampled data.


### Testing

To test a network, run 

```bash
python train_network.py -l log_loc
```
where log_loc is the location of the saved network. For instance,

```bash
python test_network.py -l saved/logs/cpu/default/
```

runs the test script on the model saved in `saved/logs/cpu/default/` and saves the results in that directory.



### Data
The code is designed to train on the fastMRI knee or brain dataset, which can be downloaded [here](https://fastmri.org/), 
or the low-field M4Raw dataset, which can be downloaded [here](https://zenodo.org/records/8056074). The path of the data should be
given in the configuration file. We have included the necessary fastMRI code in this package.

## Contact

If you have any questions/comments, please feel free to contact Charles
(Charlie) Millard at [charles.millard@ndcn.ox.ac.uk](charles.millard@ndcn.ox.ac.uk) or Mark Chiew at
[mark.chiew@utoronto.ca](mark.chiew@utoronto.ca)

## Citations
If you use this code, please cite our articles:
```
@article{millard2023theoretical,
  author={Millard, Charles and Chiew, Mark},
  journal={IEEE Transactions on Computational Imaging}, 
  title={A Theoretical Framework for Self-Supervised MR Image Reconstruction Using Sub-Sampling via Variable Density Noisier2Noise}, 
  year={2023},
  volume={9},
  number={},
  pages={707-720},
  doi={10.1109/TCI.2023.3299212}}

@misc{millard2023clean,
      title={Clean self-supervised MRI reconstruction from noisy, sub-sampled training data with Robust SSDU}, 
      author={Charles Millard and Mark Chiew},
      year={2023},
      eprint={2210.01696},
      archivePrefix={arXiv},
      primaryClass={eess.IV}}
```

## Copyright and Licensing

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License can be found in the file GNU_General_Public_License,
and is also availabe [here](https://www.gnu.org/licenses/).