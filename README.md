# Explainable Multi-Objective Model Selection for Time Series Forecasting

Code and results for the associated research paper, accepted for publication at [KDD 2024](https://kdd2024.kdd.org/), preprint available [here](https://arxiv.org/abs/2312.13038). Check out our promotional pitch video:

[![Promotional video](https://img.youtube.com/vi/utfpJNdpsRc/0.jpg)](https://www.youtube.com/watch?v=utfpJNdpsRc)

To investigate the results you can use our publicly available [Exploration tool](https://xpcr.onrender.com), so no code needs to be run on your machine (note that results might slightly differ from the preprint paper).

## Structure
All experiments were executed with the top-level Python scripts.
Our work-in-progress library for [sustainable and trustworthy reporting](https://github.com/raphischer/strep) resides within in a [separate folder](./strep/), and implements the relative index scaling, among other parts.
The [experiment logs](./results/) and [paper results](./paper_results/) also have their own folders.
Several additional `.json` and `.csv` files contain information on data, properties, and DNN models.

## Installation
All necessary libraries for running the exploration tool and DNN experiments locally can be installed via the [requirements](./requirements.txt).
For performing the baseline comparisons, you should create a second Python environment, and after installing our requirements, add the following libraries:
```
autogluon==1.0.0
autokeras==v1.1.0
tensorflow==2.14.0
auto-sklearn==0.15.0
```
If you encounter problems with `autosklearn`, you might want to adjust with the `'OPENBLAS_NUM_THREADS'` variable (see our [run.py](./run.py) for an example.)

## Usage
Firstly, you can [run our app](./run_app.py) or re-generate the [paper results](./run_paper_evaluation.py) locally.
AutoXPCR can be run on the pre-computed [property database](./results/logs.pkl) via the [corresponding script](.run_autoxpcr.py).
You can also locally assemble the property database by repeating our DNN experiments - either [all](./run_all.sh) or [single configurations](./run.py) (just pass the chosen method and data set via command line).
We included a [script to download the Monash TS data](./zenodo_forecasting_bulk_download.py), you just need to update it to use your own [access_token](https://developers.zenodo.org/#rest-api).
If facing problems with profiling, please refer to the `CodeCarbon` info pages.
A folder is created for each experiment run, and can be [merged](./run_log_processing.py) into more compact `.json` and `pandas dataframe` format, as given in the [experiment logs](./results/).
For extracting the properties and assembling the final database, `strep` uses [a special script with user-defined functions](./properties.py).

The prodedure for reproducing the results is the following:
1. Install software (e.g., via `pip install -r requirements.txt`)
2. Download [data sets](./zenodo_forecasting_bulk_download.py)
3. Run experiments via the [run script](./run.py), with all desired configurations and a specified `output-dir`
4. [Merge results](./run_log_processing.py) from the `output-dir` to a directory with merged `.json` files and database
5. Run AutoXPCR / meta-learning via the [corresponding script](./run_autoxpcr.py)
6. Explore results via our [exploration tool](./run_app_.py)

## Issues
If there are any issues, feel free to contect the paper authors.

## Terms of Use
Copyright (c) 2024 Raphael Fischer