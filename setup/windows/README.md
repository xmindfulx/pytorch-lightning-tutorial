## Installation Guide

1) **Download the repo** ( zip or `git clone` )

2) **Prepare folders** 
- Results Folder : location where the results will be stored 
- Dataset Folder : location where data files are found ( initially located inside repo )

3) [Download and setup Anaconda](https://docs.anaconda.com/anaconda/install/windows/)

4) Open anaconda prompt and install pytorch-lightning and opencv using following commands

- `pip install pytorch-lightning`
- `pip install opencv-python`

5) Running Examples - Jupyter Notebooks

- Open jupyter notebooks ( GUI application ) and navigate to downloaded repo
- Read documentation in notebooks. 
- Change `path_results` and `path_dataset` parameters to match your own prepared folders ( see step 2 ). 
- Run the notebooks ( Restart kernel and run all cells )

7) Open anaconda prompt and run Tensorboard

- `tensorboard --logdir .`

