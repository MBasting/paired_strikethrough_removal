# Paired Image to Image Translation for Strikethrough Removal From Handwritten Words

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/471974467.svg)](https://zenodo.org/badge/latestdoi/471974467)


### [Raphaela Heil](mailto:raphaela.heil@it.uu.se):envelope:, [Ekta Vats](ekta.vats@it.uu.se) and [Anders Hast](anders.hast@it.uu.se)

Code for the [DAS 2022](https://das2022.univ-lr.fr/) paper **"Paired Image to Image Translation for Strikethrough Removal From Handwritten Words"**

## Table of Contents
1. [Code](#code)
    1. [Train](#train)
    2. [Test](#test)
2. [Data](#data)
3. [Citation](#citation)
4. [Acknowledgements](#acknowledgements)


## 1 Code

### 1.1 Train
```bash
python -m src.train -file <path to config file> -section <section name>
```


### 1.2 Test
```bash
python -m src.test -file <path to config file> -data <path to test data>
```

If you want to use a checkpoint with a different name than `best_fmeasure.pth` add: `-checkpoint <filename>` and if you want to save the model outputs, i.e. cleaned images, add the flag `-save`

## Data
- IAM<sub>synth</sub>: Synthetic strikethrough dataset
    - Zenodo: [https://doi.org/10.5281/zenodo.4767094](https://doi.org/10.5281/zenodo.4767094)
    - based on the [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) database
    - multi-writer
    - generated using [https://doi.org/10.5281/zenodo.4767062](https://doi.org/10.5281/zenodo.4767062)
- Dracula<sub>real</sub>: Genuine strikethrough dataset
    - Zenodo: [https://doi.org/10.5281/zenodo.4765062](https://doi.org/10.5281/zenodo.4765062)
    - single-writer
    - blue ballpoint pen
    - clean and struck word images registered based on:
        >J. Öfverstedt, J. Lindblad and N. Sladoje, "Fast and Robust Symmetric Image Registration Based on Distances Combining Intensity and Spatial Information," in IEEE Transactions on Image Processing, vol. 28, no. 7, pp. 3584-3597, July 2019, doi: 10.1109/TIP.2019.2899947.
    ([Paper](https://ieeexplore.ieee.org/document/8643403), [Code](https://github.com/MIDA-group/py_alpha_amd_release))
- Dracula<sub>synth</sub>: Synthetic single-write dataset
    - Zenodo: [https://doi.org/10.5281/zenodo.6406538](https://doi.org/10.5281/zenodo.6406538)  
    - based on the train split of Dracula<sub>real</sub>
    - five partitions with different strikethrough strokes applied to each word


## 3 Citation
[DAS 2022](https://das2022.univ-lr.fr/)

```
@INPROCEEDINGS{heil2022strikethrough,
  author={Heil, Raphaela and Vats, Ekta and Hast, Anders},
  booktitle={15TH IAPR INTERNATIONAL WORKSHOP ON DOCUMENT ANALYSIS SYSTEMS (DAS 2022)},
  title={{Paired Image to Image Translation for Strikethrough Removal From Handwritten Words}},
  year={2022},
  pubstate={to appear}}
```

## 4 Acknowledgements 
The computations were enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE) partially funded by the Swedish Research Council through grant agreement no. 2018-05973. This work is partially supported by Riksbankens Jubileumsfond (RJ) (Dnr P19-0103:1).
