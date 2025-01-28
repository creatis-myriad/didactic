<div align="center">

# DIDACTIC

Welcome to the code repository for projects related to the *Deep manIfolD leArning CharacTerization In eChocardiography* (DIDACTIC) project.

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI: Code Format](https://github.com/creatis-myriad/didactic/actions/workflows/code-format.yml/badge.svg?branch=main)](https://github.com/creatis-myriad/didactic/actions/workflows/code-format.yml?query=branch%3Amain)

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/creatis-myriad/didactic/blob/dev/LICENSE)

## Publications

</div>

## Description
This is a project that aims to i) extract features used to evaluate cardiac function from echocardiography sequences,
and ii) used these features to perform manifold learning on a population to characterize heart diseases.

To help you follow along with the organization of the repository, here is a summary of each major package's purpose
(in alphabetical order):
```
├── didactic    <- Package directory
│   │
│   ├── apps        <- scripts to launch interactive applications for data visualization, model exploration, etc.
│   │    └── representation_visualization.py    <- application to visualize the distribution of the dataset in a
│   │                                              representation learned by a model, w.r.t. dataset variables.
│   │
│   ├── config      <- configuration files for Hydra, to define the options available for the `didactic` runner.
│   │   ├── experiment  <- pre-configured experiments for the `didactic` runner.
│   │   └── task        <- configuration for individual components of the full pipeline.
│   │       ├── data                    <- subsets of available data.
│   │       ├── model                   <- model components.
│   │       └── time_series_tokenizer   <- tokenizers for time-series data.
│   │
│   ├── data        <- dataset-specific code
│   │   └── cardinal    <- CARDINAL dataset-specific code.
│   │       └── predict.py    <- Implementation of a Lightning `BasePredictionWriter` to evaluate a multimodal
│   │                            representation learning model on the CARDINAL dataset.
│   │
│   ├── models      <- custom layers and modules used in our pipeline
│   │   ├── layers.py       <- layers, e.g. prediction head, ordinal classification, etc.
│   │   ├── tabular.py      <- module for tabular data tokenization (fig. 2.a in the paper).
│   │   └── time_series.py  <- module for time-series data tokenization (fig. 2.b in the paper).
│   │
│   ├── scripts     <- various scripts, e.g. to produce figures, aggregate results from multiple jobs, etc.
│   │
│   └── tasks       <- implementation of complete training and inference pipeline for specific tasks (e.g. classification,
│       │              segmentation, etc.)
│       └── cardiac_multimodal_representation.py    <- full pipeline to train/infer on tabular and time-series data
│                                                      (fig. 1 in the paper).
│
├── shell_scripts    <- shell scripts
│   └── train.sh        <- train multimodal transformers for HT severity classification on the CARDINAL dataset
│
└── vital       <- separate repository (included as a git submodule), of generic PyTorch modules, losses, metrics, and
                   other tooling (e.g. image processing) commonly used. Also contains data-handling for specialized
                   medical imaging datasets, e.g. CAMUS, CARDINAL.
```

## How to Run

### Install
First, download the project's code:
```shell script
# clone project
git clone --recurse-submodules https://github.com/creatis-myriad/didactic.git
```
Next you have to install the project and its dependencies. The project's dependency management and packaging is handled
by [`poetry`](https://python-poetry.org/) so the recommended way to install the project is in a virtual environment
(managed by your favorite tool, e.g. `conda`, `virtualenv`, `poetry`, etc.), where
[`poetry` is installed](https://python-poetry.org/docs/#installation). That way, you can simply run the command:
```shell script
poetry install
```
from the project's root directory to install it in editable mode, along with its regular and development dependencies.
This command also takes care of installing the local `vital` submodule dependency in editable mode, so that you can
edit the library and your modifications will be automatically taken into account in your virtual environment.

> **Note**
> When a [`poetry.lock`](poetry.lock) file is available in the repository, `poetry install` will automatically use it to
> determine the versions of the packages to install, instead of resolving anew the dependencies in `pyproject.toml`.
> When no `poetry.lock` file is available, the dependencies are resolved from those listed in `pyproject.toml`, and a
> `poetry.lock` is generated automatically as a result.

> **Warning**
> Out-of-the-box, `poetry` offers flexibility on how to install projects. Packages are natively `pip`-installable just
> as with a traditional `setup.py` by simply running `pip install <package>`. However, we recommend using `poetry`
> because of an [issue with `pip`-installing projects with relative path dependencies](https://github.com/python-poetry/poetry/issues/5273)
> (the `vital` submodule is specified using a relative path). When the linked issue gets fixed, the setup instructions
> will be updated to mention the possibility of using `pip install .`, if one wishes to avoid using `poetry` entirely.

To test that the project was installed successfully, you can try the following command from the Python REPL:
```python
# now you can do:
from didactic import Whatever
```

> **Warning**
> All following commands in this README (and other READMEs for specific packages), will assume you're working from
> inside the virtual environment where the project is installed.

### Data
Next, navigate to the data folder for the
[CARDINAL](https://github.com/creatis-myriad/vital/tree/dev/vital/data/cardinal) dataset and follow the [instructions
on how to download and prepare the data](https://github.com/creatis-myriad/vital/tree/dev/vital/data/cardinal/README.md).

### Configuring a Run
This project uses Hydra to handle the configuration of the
[`didactic` runner script](didactic/runner.py). To understand how to use Hydra's CLI, refer to its
[documentation](https://hydra.cc/docs/intro/). For this particular project, preset configurations for various parts of
the `didactic` runner pipeline are available in the [config package](didactic/config). These files are meant to be
composed together by Hydra to produce a complete configuration for a run.

#### Reproduce experiments from the paper
To reproduce the experiments described in the paper, you can use the [provided shell script](shell_scripts/train.sh).
Comments indicate which experiments each command corresponds to.

To run the shell script, you have to provide it with 3 arguments:
1. The path to the dataset root folder;
2. The path to the directory where the results will be saved;
3. The name of the Comet project to log the experiments to.

For example:
```shell script
./shell_scripts/train.sh /path/to/dataset /path/to/results comet-project-name
```

#### Configure your own experiments
If you want to run other experiments than the ones provided in the [shell script](shell_scripts/train.sh) (and reported
in the paper), you can take inspiration from the commands there and from the pre-configured Hydra experiments in the
[config/experiment](didactic/config/experiment) folder.

Below are some basic examples of how to customize experiments using the Hydra CLI:
```shell script
# list generic trainer options and datasets on which you can train
didactic-runner -h

# select high-level options of task to run, and architecture and dataset to use
didactic-runner task=<TASK> task/model=<MODEL> data=<DATASET>

# run one of the fully pre-configured 'experiment' from the `config/experiment` folder (e.g. XTab)
didactic-runner +experiment=xtab
```

To create your own pre-configured experiments, like the one used in the last example, we refer you to [Hydra's own
documentation on configuring experiments](https://hydra.cc/docs/patterns/configuring_experiments/).

### Tracking experiments
By default, Lightning logs runs locally in a format interpretable by
[Tensorboard](https://www.tensorflow.org/tensorboard/).

Another option is to use [Comet](https://www.comet.ml/) to log experiments, either online or offline. To enable the
tracking of experiments using Comet, simply use one of the pre-built Hydra configuration for Comet. The default
configuration is for Comet in `online` mode, but you can use it in `offline` mode by selecting the corresponding config
file when launching the [`didactic` runner script](didactic/runner.py):
```bash
didactic-runner logger=comet/offline ...
```
To configure the Comet API and experiment's metadata, Comet relies on either i) environment variables (which you can set
in a `.env` that will automatically be loaded using `python-dotenv`) or ii) a `.comet.config` file. For
more information on how to configure Comet using environment variables or the config file, refer to
[Comet's configuration variables documentation](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables).

An example of a `.comet.config` file, with the appropriate fields to track experiments online, can be found
[here](https://github.com/creatis-myriad/vital/tree/dev/.comet.config). You can simply copy the file to the directory
of your choice within your project (be sure not to commit your Comet API key!!!) and fill the values with your own Comet
credentials and workspace setup.

> **Note**
> No change to the code is necessary to change how the `CometLogger` handles the configuration from the `.comet.config`
> file. The code simply reads the content of the `[comet]` section of the file and uses it to create a `CometLogger`
> instance. That way, you simply have to ensure that the fields present in your configuration match the behavior you
> want from the `CometLogger` integration in Lighting, and you're good to go!


## How to Contribute

### Environment Setup
When installing the dependencies using `poetry install` as [described above](#install), the resulting environment is
already fully configured to start contributing to the project. There's nothing to change to get coding!

### Version Control Hooks
Before first trying to commit to the project, it is important to setup the version control hooks, so that commits
respect the coding standards in place for the project. The [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file
defines the pre-commit hooks that should be installed in any project contributing to the `vital` repository. To setup
the version control hooks, run the following command:
```shell script
pre-commit install
```

> **Note**
> In case you want to copy the pre-commit hooks configuration to your own project, you're welcome to :)
> The configuration for each hook is located in the following files:
> - [isort](https://github.com/timothycrosley/isort): [`pyproject.toml`](./pyproject.toml), `[tool.isort]` section
> - [black](https://github.com/psf/black): [`pyproject.toml`](./pyproject.toml), `[tool.black]` section
> - [flake8](https://gitlab.com/pycqa/flake8): [`setup.cfg`](./setup.cfg), `[flake8]` section
>
> However, be advised that `isort` must be configured slightly differently in each project. The `src_paths` tag
> should thus reflect the package directory name of the current project, in place of `vital`.


## References
If you find this code useful, please consider citing the paper implemented in this repository relevant to you from the
list below:
```bibtex

```
