## Installation Guide

1) Download the repo ( zip or `git clone` )

2) Prepare folders - Dataset ( data files found inside repo ), Results

3) [Download and setup Anaconda](https://docs.anaconda.com/anaconda/install/windows/)

4) Open conda command prompt and install pytorch-lightning and opencv

- `pip install pytorch-lightning`
- `pip install opencv`

5) Running Examples Via Jupyter Notebooks

- Open jupyter notebooks ( GUI application ) and navigate to downloaded repo
- Read documentation in notebooks. 
- Change `path_results` and `path_dataset` parameters to match your own prepared folders ( see step 2 ). 
- Run the notebooks ( Restart kernel and clear output, Restart kernel and run all )

7) Open conda command prompt and run Tensorboard

- `tensorboard --logdir .`

