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

Create the data/ - directory for storing the dataset used in this task. This directory will be ignored by git.

```
$ mkdir data/
```

## Run the notebooks to reproduce the results

Open jupyter notebooks on your default browser by running the command in the project directory (ml_project/):

`$ jupyter-notebook`

Your default browser opens a new tab with a view of your project files. Open a notebook by clicking it's name.

Once the notebook is open, you can run the results by selecting Cell -> Run Cells from the menu bar at the top.
