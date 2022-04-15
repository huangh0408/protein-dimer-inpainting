# protein-dimer-inpainting

## Installation

Clone this repo.
```bash
git clone https://github.com/huangh0408/protein-dimer-inpainting.git
```

Prerequisites
* Python 2.7
* tensorflow-gpu 1.12.0
* ipdb
* opencv-python
* glob
* cPickle


## Datasets

### train set and validation set

We use [3dcomplex](https://shmoo.weizmann.ac.il/elevy/3dcomplexV6/Home.cgi) and [masif](https://github.com/LPDI-EPFL/masif) datasets.

### Independent test set

We use [gremlin](https://shmoo.weizmann.ac.il/elevy/3dcomplexV6/Home.cgi), [evcomplex](https://evcouplings.org/),[benchmark 5.0](https://zlab.umassmed.edu/benchmark/) and [casp-capri](https://predictioncenter.org/download_area/) datasets.

### Your own datasets
Please prepare the pdb file and the corresponding chain file, which are required by our scripts to generate inter-protein and intra-protein contact/distance map.

```bash
# To generate the whole contact/distance map.
cd generate_contact_map
bash work.sh
```


## Training New Models

### on our datasets

There are three folders to present  three kinds of datasets respectively. You can download the data [here](ftp:/202.112.126.139/protein-dimer-inpainting). 

```bash
# To train on the dataset. Notice that you should modify the input file directory and checkpoint directory in the work_train.sh file.
bash work_train.sh
```

### on your own datasets

```bash
# To train on the you dataset, for example.
python train.py --input_dir[the path of original images] --mode=[contact distance slice] --netsize[128 256 512] 
```
There are many options you can specify. Please use `python train.py --help` or see the options



## Pre-trained weights and test model

There are three folders to present pre-trained for three kinds of datasets respectively. You can download the pre-trained model [here](ftp:/202.112.126.139/protein-dimer-inpainting). 

### testing

```bash
# To test on the dataset. Notice that you should modify the test set directory and checkpoint directory in the work_test.sh file.
bash work_test.sh
```

## Evaluation

We calculate the precision,which is defined as TP/N. Such as Top 5, 10, 20, L/10, L/5, L/2, L used in the intra-protein contact map prediction. For the overall results, we calculate the mean precision. Additionaly, we calculate the success rate, which is defined the percentage of the targets with at least one successfully predicted contact when a certain number of predicted contacts are considered, compared to all the targets in the test set.

### Precision & Success rate

```bash
# To evaluate on the dataset. Notice that you should modify the output file directory and groundtruth file directory in th work_evaluate.sh file.
bash work_evaluate.sh
```

### Different Version

#### scripts_version1.0 
   original scripts without mask
#### scripts_version2.0 
   scripts with region mask
#### scripts_version3.0 
   implement in environment python 3.7
#### scripts_demo 
   jupyter-notebook to test our model

### Citation
If you use this code for your research, please cite our papers.
```
@article{huang2021,
  title={Inter-protein contact map generated only from intra-monomer by image inpainting},
  author={He Huang, Chengshi Zeng, Xinqi Gong},
  booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2021},
  doi={10.1109/BIBM52615.2021.9669709}
}
```

## Acknowledgments

Our inpainting codes refer to [Inpainting](https://github.com/jazzsaxmafia/Inpainting) and the readme.md file refers to [Rethinking-Inpainting-MEDFE](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE).

## Contacts


