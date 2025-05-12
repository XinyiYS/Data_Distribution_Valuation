# Data Distribution Valuation
Official implementation of our NeurIPS 2024 paper "Data Distribution Valuation" in Advances in Neural Information Processing Systems 37: 38th Annual Conference on Neural Information Processing Systems (**NeurIPS'24**): **Xinyi Xu**, Shuaiqi Wang, Chuan-Sheng Foo, Bryan Kian Hsiang Low and Giulia Fanti. |[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/04b98fd38bd42810d0764cb6c46d10d8-Abstract-Conference.html)|[poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/96892.png?t=1731176029.7581189)|[talk](https://neurips.cc/virtual/2024/poster/96892)|

# Preparations

## Packages:

Recommended to use anaconda/miniconda to manage the environment, the requried packages are in the `environment.yml`.

## Datasets:

Due to the size restrictions of the files/directories on Github, the actual datasets are _not_ directly uploaded. Most image datasets are constructed/used via the methods from torchvision (so make sure to install torchvision).

The datasets for regression are from Kaggle： [California housing](https://www.kaggle.com/datasets/camnugent/california-housing-prices), [Kings housing sales](https://www.kaggle.com/harlfoxem/housesalesprediction), and [US Census](https://www.kaggle.com/datasets/census/census-bureau-usa).


# Usage

There are a few main python scripts that implement the logic of accepting arguments, constructing datasets, training an ML model (if applicable), running our methods and saving the results. For example `run_Ours.py` and `run_Ours_conditional.py` and in the directory `regression/` there are `` 

To run experiments conveniently with a list of arguments, take a look at the bash scripts, such as `sh_run_experiments.sh`.

Running the python or bash scripts will create a `results` directory where the results will be saved.

## Citing
If you find our research useful, please consider citing our work.
```
 @inproceedings{xu2024datadistributionvaluation,
                title={Data Distribution Valuation}, 
                booktitle={Thirty-eighth Conference on Neural Information Processing Systems}, 
                author={Xinyi Xu and Shuaiqi Wang and Chuan-Sheng Foo and Bryan Kian Hsiang Low and Giulia Fanti},
                year={2024}}
```
