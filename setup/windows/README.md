## Common Windows 10 Problems

A few users have reported different jupyter notebook errors in the form of kernel crashes / freezes, or errors that are generally .dll file related. An example of such is  

`Initializing libiomp5md.dll, but found mk2iomp5md.dll already initialized`

A temporary solution is to include the following lines as the first cell in your jupyter notebook.

`import os`

`os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"`

However, this solution is not perfect, and sensitvity tests haven't been done to see exactly what this impacts. 

The safer and more stable option would be to invest time learning the basics of [docker containers](https://www.docker.com/resources/what-container).

## Windows 10 Setup Guide

1) **Download the repo** ( zip or `git clone` )

2) **Follow Beginner or Intermediate Setup Guide**
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
