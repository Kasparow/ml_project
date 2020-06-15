# ML course individual project track

This repo is an individual project assignment for machine learning course at the University of Helsinki. Following basic algorithms are implemented and discussed in the notebooks in this repo:
- Perceptron and the assumption of linearity with linear activations
- Probabilistic modeling with Multinomial Naïve Bayes classifier
- Dimensionality reduction for data visualization with Principal Component Analysis (and T-SNE)

## Requirements

- Python 3.6+
- Datasets used in the tasks placed in ./data/ directory


## Setup environment

### Clone repository

```
$ git clone https://github.com/Kasparow/ml_project.git
$ cd ml_project/
```

### Install dependencies

We will use a python virtual environment for managing dependencies. Virtual environment named venv/ is added is ignored by git. Create and activate a virtual environment:

```
$ mkdir venv
$ python3 -m venv venv/
$ source venv/bin/activate
```

Install project dependencies:

```
$ pip install -r requirements.txt
```

### Download data

Download all of the data used in the project from the following address:

- https://drive.google.com/open?id=1lAWfZmuDcbMvYJ6hAD9yg1Jd-PkFL2BD

And unzip the data.zip directory under the project root: ml_project/data/. The data-folder will be ignored by git.


Alternatively, create an empty data/ - directory for storing your own datasets.

```
$ mkdir data/
```

## Run the notebooks to reproduce the results

Open jupyter notebooks on your default browser by running the command in the project directory (ml_project/):

```
$ jupyter-notebook
```

Your default browser opens a new tab with a view of your project files. Open a notebook by clicking it's name.

Once the notebook is open, you can run the results by selecting Cell -> Run Cells from the menu bar at the top.
