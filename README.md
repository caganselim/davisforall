# DAVIS For All - A Generic J&F Evaluation Package for UVOS

This package is a modified version of DAVIS evaluation toolkit which is used to evaluate semi-supervised and unsupervised video multi-object segmentation models for the <a href="https://davischallenge.org/davis2017/code.html" target="_blank">DAVIS 2017</a> to evaluate **ANY** VOS dataset in the UVOS setting.


## How to Evaluate?

In order to evaluate your unsupervised method in any dataset, execute the following command substituting `results/unsupervised/rvos` by the folder path that contains your results:
```bash
python evaluation_method.py
```

You can either pass custom parameters or *default_dataset_path* and *default_results_path* in the code. 

This code also allows you to pass different folder names for the masks and image folders as well.


## Citation

Please cite both papers in your publications if DAVIS or this code helps your research.

```latex
@article{Caelles_arXiv_2019,
  author = {Sergi Caelles and Jordi Pont-Tuset and Federico Perazzi and Alberto Montes and Kevis-Kokitsi Maninis and Luc {Van Gool}},
  title = {The 2019 DAVIS Challenge on VOS: Unsupervised Multi-Object Segmentation},
  journal = {arXiv},
  year = {2019}
}
```

```latex
@article{Pont-Tuset_arXiv_2017,
  author = {Jordi Pont-Tuset and Federico Perazzi and Sergi Caelles and Pablo Arbel\'aez and Alexander Sorkine-Hornung and Luc {Van Gool}},
  title = {The 2017 DAVIS Challenge on Video Object Segmentation},
  journal = {arXiv:1704.00675},
  year = {2017}
}
```

