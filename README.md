# Explainable Multi-Objective Model Selection for Time Series Forecasting

Code and results for the associated research paper (currently under review), this code is anonymously published for reviewers.

## Structure
All paper-specific experiments were executed with the top-level Python scripts.
Our work-in-progress [generalized library for ML properties](./mlprops/) resides within in a separate folder.
The [experiment logs](./results/) and [paper results](./paper_results/) also have their own folders.
The meta files (.json) contain meta information on data, properties, models and the environment.

## Installation
All code was executed with Python 3.7, GluonTS 0.11.8 and Scikit-learn 1.0.20 on Ubuntu 22.04.2, please refer to [requirements](./requirements.txt) for all necessary dependencies.
You can get even more detailed information on the exact setup from the [logs](./results/merged_new03/).
Depending on how you intend to use this software, only some packages are required.

## Usage
To investigate the results you can use our publicly available [Exploration tool](http://167.99.254.41/), so no code needs to be run on your machine.

You can also [run the evaluation](./run_evaluation.py) locally and pass different modes: generate paper results, start the interactive app, run the meta learning, etc.
All [paper results](./paper_results/) (plots and tables) were generated via this script.

New experiments can also be executed with the designated [script](run.py) - pass the chosen method, software backend and more configuration options via command line.
We included a [script to download the Monash TS data](./zenodo_forecasting_bulk_download.py).
If facing problems with profiling, refer to the `CodeCarbon` info pages.
A folder is created for each experiment run, and can be [merged](./parse_logs.py) into more compact `.json` and `pandas dataframe` format representing our "database".
For extracting the properties, mlprops uses [a script with user-defined functions](./properties.py).

The prodedure for reproducing the results is the following:
1. Install software
2. [Download data sets](./zenodo_forecasting_bulk_download.py)
3. Run desired experiments via calling the [run script](./run.py) with all configurations and a specified `output-dir`
4. [Merge results](./parse_logs.py) from the `output-dir` to a directory with merged `.json` files and database
5. Run meta learning by calling the [evaluation script](./run_evaluation.py) and passing `--mode meta`
6. Explore results via [evaluation script](./run_evaluation.py) and `--mode interactive`

## Terms of Use
Copyright (c) 2023 authors of paper under review.