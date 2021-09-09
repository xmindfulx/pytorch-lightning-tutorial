## Windows 10 Setup Guide

1) **Download the repo** ( zip or `git clone` )

2) **Follow Beginner or Intermediate Setup Environment Guide**
- Choose Beginner if you are not comfortable with terminal / command line
- Choose Intermediate if you are comfortable with terminal / command line

### Beginner Guide 

1) [Download and setup Anaconda](https://docs.anaconda.com/anaconda/install/windows/)

2) **Open anaconda prompt and install pytorch-lightning and opencv using following commands**

- `pip install pytorch-lightning`
- `pip install opencv-python`

3) **Running Examples - Jupyter Notebooks**

- Open jupyter notebooks ( GUI application ) and navigate to downloaded repo
- Read documentation in notebooks. 
- Run the notebooks ( Restart kernel and run all cells )

### Intermediate Guide

1) [Download and setup Miniconda](https://docs.conda.io/en/latest/miniconda.html)

3) **Build New Virtual Environment**

- `conda create -n tutorial -y`
- `conda activate tutorial`
- `conda install -c conda-forge opencv -y`
- `conda install -c conda-forge matplotlib ipython jupyter -y`
- `pip install pytorch-lightning`

3) **Running Examples - Jupyter Notebooks**

- start notebook by `jupyter-notebook .`
- Read documentation in notebooks. 
- Run the notebooks ( Restart kernel and run all cells )
