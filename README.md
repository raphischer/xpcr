# Explainable Multi-Objective Model Selection for Time Series Forecasting

Code and results for the associated research paper (currently under review).
To investigate the results you can use our publicly available [Exploration tool](http://167.99.254.41/), so no code needs to be run on your machine (note that single results might slightly differ from the submitted paper, which will be updated upon acceptance).

## Structure
All paper-specific experiments were executed with the top-level Python scripts.
Our work-in-progress [generalized library for ML properties](./mlprops/) resides within in a separate folder, and implements the relative index scaling, among other parts.
The [experiment logs](./results/) and [paper results](./paper_results/) also have their own folders.
Several additional `.json` and `.csv` files contain information on data, properties, models and the environment.

## Installation
All necessary libraries for running the exploration tool locally can be installed via the [requirements](./requirements.txt).
For performing the ML experiments, detailed information on environment and libraries is given in the [merged logs](./results/merged_dnns/).

## Usage
You can [run our evaluation](./run_evaluation.py) locally and pass different modes: start the interactive app (default), generate paper results, run the meta learning, etc.
All [paper results](./paper_results/) (plots and tables) were generated via this script.

The ML experiments can be executed with the [designated script](run.py) - pass the chosen method, software backend and more configuration options via command line.
We included a [script to download the Monash TS data](./zenodo_forecasting_bulk_download.py).
If facing problems with profiling, refer to the `CodeCarbon` info pages.
A folder is created for each experiment run, and can be [merged](./parse_logs.py) into more compact `.json` and `pandas dataframe` format, as given in the [experiment logs](./results/).
For extracting the properties, `mlprops` uses [a special script with user-defined functions](./properties.py).

The prodedure for reproducing the results is the following:
1. Install software
2. [Download data sets](./zenodo_forecasting_bulk_download.py)
3. Run desired experiments via calling the [run script](./run.py) with all configurations and a specified `output-dir`
4. [Merge results](./parse_logs.py) from the `output-dir` to a directory with merged `.json` files and database
5. Run meta learning by calling the [evaluation script](./run_evaluation.py) and passing `--mode meta`
6. Explore results via [evaluation script](./run_evaluation.py) and `--mode interactive`

## Terms of Use
Copyright (c) 2024 tmplxz