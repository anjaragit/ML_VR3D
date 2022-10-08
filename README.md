
# Machine Learning VR3D Project #

## INTRO
c'est projet a pour but d'optimiser ou de reduire le temps consommer par un dessinateur 3D ou 2D pour trace ou convertir 
un plan 2D en 3D et vice versa.Grace a l'utilisation de la Deep Learning on ait possible de faire avec de erreur le plus 
petit que faite par un etre humain .Donc a partir de maintenant , juste en une seule clique bouton et attend quelque 
minute , vous avez fait le travail de plus de dizaine de dessineur.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://camo.githubusercontent.com/34b3a249cd6502d0a521ab2f42c8830b7cfd03fa/687474703a2f2f7777772e6d7970792d6c616e672e6f72672f7374617469632f6d7970795f62616467652e737667)](http://mypy-lang.org/)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![made-with-bash](https://img.shields.io/badge/Made%20with-Bash-1f425f.svg)](https://www.gnu.org/software/bash/)

## Project structure

```
├── config                       <- Directory containing configuration files
├── README.md                 <- The top-level README for developers using this project.
├── dataset                   <- Sample data from different sources for unit and integration tests
│   ├── LasFile              <- Data filter
│   └── PointCloud                   <- The original data.
│
├── docs               
│
├── models                    <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks                 <- Jupyter notebooks. 
│
├── reports                   <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures               <- Generated graphics and figures to be used in reporting
│
├── requirements.txt          <- The requirements 
│                         
│
├── setup.py                  <- Make this project pip installable with `pip install -e`
├── tests                     <- The pytest tests root directory for unit / integration / e2e tests
├── src/ML_VR3D                
│   ├── __init__.py    
│   │
│   ├── util                  <- Scripts with common processing
│   │
│   ├── data                  <- Scripts to download or generate data
|   |   ├── __init__.py    
│   │   └── common 
|   |   |   |_ Labelisation.py          <- script labelize automaticaly data
│   │   └── H5_file             <- Transform xyz value to H5 and extracte new feature       
│   │
│   │
│   ├── iamodels              <- Scripts to train IA models and then use trained models to make
│   │   │                       predictions
|   |   ├── __init__.py    
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization         <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```


## Development

### Prerequisites
All you need is the following configuration already installed:

* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [pyenv prerequisites for ubuntu](https://github.com/pyenv/pyenv/wiki#suggested-build-environment). Check the prerequisites for your OS.

```
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

* `pyenv` installed and available in path [pyenv installation](https://github.com/pyenv/pyenv#installation) with Prerequisites 
* For tests/integration tests with mongoDB, you need [Docker](https://docs.docker.com/engine/install/ubuntu/)



### Add format, lint code tools

#### Autolint/Format code with Black in IDE:

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

* Auto format via IDE https://github.com/psf/black#pycharmintellij-idea
* [Optional] You could setup a pre-commit to enforce Black format before commit https://github.com/psf/black#version-control-integration
* Or remember to type `black .` to apply the black rules formatting to all sources before commit
* [Jenkins](https://jenkins.parts-advisor.com/job/parts-io/job/etl/lastCompletedBuild/testReport/) will complain and tests will fail if black format is not applied

* Add same mypy option for vscode in `Preferences: Open User Settings`
* Use the option to lint/format with black and flake8 on editor save in vscode

#### Checked optional type with Mypy [PEP 484](https://www.python.org/dev/peps/pep-0484/)

[![Checked with mypy](https://camo.githubusercontent.com/34b3a249cd6502d0a521ab2f42c8830b7cfd03fa/687474703a2f2f7777772e6d7970792d6c616e672e6f72672f7374617469632f6d7970795f62616467652e737667)](http://mypy-lang.org/)

Configure Mypy to help annotating/hinting type with Python Code. It's very useful for IDE and for catching errors/bugs early. 

* Install [mypy plugin for intellij](https://plugins.jetbrains.com/plugin/11086-mypy)
* Adjust the plugin with the following options:
    ```
    "--follow-imports=silent",
    "--show-column-numbers",
    "--ignore-missing-imports",
    "--disallow-untyped-defs",
    "--check-untyped-defs"
    ```  
* Work in Progress to adjust Mypy preferences
* Documentation: [Type hints cheat sheet (Python 3)](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
* Add same mypy option for vscode in `Preferences: Open User Settings` 

#### Install Sonarlint plugin

Detect Code Quality and Security issues on the fly

* [Use SonarLint in IntelliJ IDEA](https://www.sonarlint.org/intellij/)
* [Use SonarLint in vscode](https://www.sonarlint.org/vscode)

#### Isort

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

* [isort is the default on pycharm](https://www.jetbrains.com/help/pycharm/code-style-python.html#imports_table)
* [isort with vscode](https://cereblanco.medium.com/setup-black-and-isort-in-vscode-514804590bf9)
* Lint/format/sort import on save with vscode in `Preferences: Open User Settings`:

```
{
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

* isort configuration for pycharm. See [Set isort and black formatting code in pycharm](https://www.programmersought.com/article/23126972182/)
* You can use `make lint` command to check flake8/mypy rules & apply automatically format black and isort to the code with the previous configuration

```
isort . --virtual-env dspioenv
```

## PROJECT SCOPE
Le model de base de ce project et le pointnet, visible sur  [Github Pointnet](https://github.com/charlesq34/pointnet)

## Exemple and Integration
![](https://github.com/anjaragit/ML_VR3D/blob/main/MAS.gif)
