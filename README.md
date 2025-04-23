# AutoXPCR - Explainable Multi-Objective Model Selection for Time Series Forecasting

Code and results for our associated research paper, published and presented at [KDD 2024 (open access)](https://dl.acm.org/doi/10.1145/3637528.3672057). Also check out our promotional pitch video:

[![Promotional video](https://img.youtube.com/vi/utfpJNdpsRc/0.jpg)](https://www.youtube.com/watch?v=utfpJNdpsRc)

To investigate the results you can use our publicly available [Exploration tool](https://strep.onrender.com/?database=XPCR), so no code needs to be run on your machine. Note that this software is [work in progress](https://github.com/raphischer/strep) and subject to change, so you might encounter delays, off-times, and slight differences to our paper.

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

## Issues and Contribution
If you experience any issues or want to contribute, feel free to reach out via the contact information in our paper!

## Citation
If you appreciate our work and code, please cite [our paper](https://dl.acm.org/doi/10.1145/3637528.3672057) as given by ACM:

Raphael Fischer and Amal Saadallah. 2024. AutoXPCR: Automated Multi-Objective Model Selection for Time Series Forecasting. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24). Association for Computing Machinery, New York, NY, USA, 806–815. https://doi.org/10.1145/3637528.3672057

or using the bibkey below:
```
@inproceedings{10.1145/3637528.3672057,
author = {Fischer, Raphael and Saadallah, Amal},
title = {AutoXPCR: Automated Multi-Objective Model Selection for Time Series Forecasting},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3672057},
doi = {10.1145/3637528.3672057},
abstract = {Automated machine learning (AutoML) streamlines the creation of ML models, but few specialized methods have approached the challenging domain of time series forecasting. Deep neural networks (DNNs) often deliver state-of-the-art predictive performance for forecasting data, however these models are also criticized for being computationally intensive black boxes. As a result, when searching for the "best" model, it is crucial to also acknowledge other aspects, such as interpretability and resource consumption. In this paper, we propose AutoXPCR - a novel method that produces DNNs for forecasting under consideration of multiple objectives in an automated and explainable fashion. Our approach leverages meta-learning to estimate any model's performance along PCR criteria, which encompass (P)redictive error, (C)omplexity, and (R)esource demand. Explainability is addressed on multiple levels, as AutoXPCR pro-vides by-product explanations of recommendations and allows to interactively control the desired PCR criteria importance and trade-offs. We demonstrate the practical feasibility AutoXPCR across 108 forecasting data sets from various domains. Notably, our method outperforms competing AutoML approaches - on average, it only requires 20\% of computation costs for recommending highly efficient models with 85\% of the empirical best quality.},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {806–815},
numpages = {10},
keywords = {automl, explainable ai, meta-learning, resource-aware ml, time series forecasting},
location = {Barcelona, Spain},
series = {KDD '24}
}
```

© Raphael Fischer
