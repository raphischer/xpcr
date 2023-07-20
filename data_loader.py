from datetime import datetime, timedelta
import re
import os
from distutils.util import strtobool

import numpy as np
import pandas as pd

from mlprops.util import fix_seed

from data_lookup_info import LOOKUP



# most of this code, except for the CUSTOM SUBSAMPLING of datasets, is taken from the original TSForecast repository
# https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py


# Seasonality values corresponding with the frequencies: minutely, 10_minutes, half_hourly, hourly, daily, weekly, monthly, quarterly and yearly
# Consider multiple seasonalities for frequencies less than daily
SEASONALITY_MAP = {
   "minutely": [1440, 10080, 525960],
   "10_minutes": [144, 1008, 52596],
   "half_hourly": [48, 336, 17532],
   "hourly": [24, 168, 8766],
   "daily": 7,
   "weekly": 365.25/7,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

# Frequencies used by GluonTS framework
FREQUENCY_MAP = {
   "minutely": "1min",
   "10_minutes": "10min",
   "half_hourly": "30min",
   "hourly": "1H",
   "daily": "1D",
   "weekly": "1W",
   "monthly": "1M",
   "quarterly": "1Q",
   "yearly": "1Y"
}

TIMEDELTA_MAP = {   
    "minutely": timedelta(minutes=1),
    "10_minutes": timedelta(minutes=10),
    "half_hourly": timedelta(minutes=30),
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
    "monthly": timedelta(days=365.25 / 12),
    "quarterly": timedelta(days=365.25 / 4),
    "yearly": timedelta(days=365.25 )
}


def subsampled_to_orig(subsampled_ds):
    match = re.match(r'(.*)_(\d+)', subsampled_ds)
    if match is None or match.group(0) != subsampled_ds:
        return subsampled_ds
    return match.group(1)


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
    ds_sample_seed=-1,
    amount_of_series=0.5,
    amount_of_length=0.5,
    ext_fc_horizon=None
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if forecast_horizon is None:
            if ext_fc_horizon is None:
                raise Exception("Please provide the required forecast horizon")
            forecast_horizon = ext_fc_horizon

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        if frequency is not None:
            freq = FREQUENCY_MAP[frequency]
            seasonality = SEASONALITY_MAP[frequency]
            if isinstance(seasonality, list):
                seasonality = min(seasonality)
            timedelta_val = TIMEDELTA_MAP[frequency]
        else:
            freq = "1Y"
            timedelta_val = TIMEDELTA_MAP['yearly']
            seasonality = 1

        if "start_timestamp" not in all_data:
            all_data["start_timestamp"] = [datetime(1900, 1, 1, 0, 0, 0) for _ in range(len(all_data['series_name']))]
        
        # check for constant series in dataset (in forecast range) and remove them
        to_remove = []
        for idx, ser in enumerate(all_series):
            for idx_s in range(len(ser) - forecast_horizon):
                # check all windows in series for contant values
                if np.unique(ser[idx_s:(idx_s + forecast_horizon)]).size == 1:
                    to_remove.append(idx)
                    break
        if len(to_remove) > 0:
            print(f'Not considering {len(to_remove)} constant series', [all_data['series_name'][idx] for idx in to_remove])
            for idx in reversed(to_remove):
                all_data['series_name'].pop(idx)
                all_data['start_timestamp'].pop(idx)
                all_series.pop(idx)

        all_data[value_column_name] = all_series

        # CUSTOM SUBSAMPLING
        if ds_sample_seed != -1:
            assert amount_of_length <= 1 and amount_of_length > 0, "please pass valid amount_of_length (0 < amt <= 1)"
            assert amount_of_series <= 1 and amount_of_series > 0, "please pass valid amount_of_series (0 < amt <= 1)"
            fix_seed(ds_sample_seed)

            # sanity check the end dates
            end_dates = []
            series_names = []
            for idx, key in enumerate(all_data['series_name']):
                start = all_data['start_timestamp'][idx]
                len_ts = len(all_data['series_value'][idx])
                end = start + timedelta_val * len_ts
                series_names.append(key)
                end_dates.append(end)
            for edate, key in zip(end_dates, series_names):
                if edate != end_dates[0]:
                    print(f'WARNING! During subsampling {full_file_path_and_name}, some time series were found to have differing end timestamps\n{key} end date is {edate}, {series_names[0]} end date is {str(end_dates[0])}')
                    break
            
            # fill new dict with sampled parts of TS data
            sampled_data = {key: [] for key in all_data.keys()}
            no_series = len(all_data['series_name'])
            selected_series = np.random.choice(np.arange(no_series), int(no_series * amount_of_series), replace=False)
            for s_idx in selected_series:
                for key in all_data.keys():
                    if key not in ['start_timestamp', 'series_value']:
                        sampled_data[key].append(all_data[key][idx])
                start, values = all_data['start_timestamp'][s_idx], all_data['series_value'][s_idx]
                ts_new_len = int(len(values) * amount_of_length)
                start_offset = np.random.randint(0, len(values) - ts_new_len, 1)[0]
                sampled_data['start_timestamp'].append(start + timedelta_val * start_offset)
                sampled_data['series_value'].append(values[start_offset:(start_offset + ts_new_len)])
                assert len(sampled_data['series_value'][-1]) == ts_new_len

            # print summary
            no_values_new = sum([len(vals) for vals in sampled_data['series_value']])
            no_values_old = sum([len(vals) for vals in all_data['series_value']])
            print(f'Sampled to {no_values_new / no_values_old:4.3f} of original data ({amount_of_series:3.2f} of series with {amount_of_length:3.2f} of their original length)!')

            loaded_data = pd.DataFrame(sampled_data)
        else:
            loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            freq,
            seasonality,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

if __name__ == '__main__':

    datadir = 'mnt_data/data'
    ds_stats = []
    ds_n_vals = []
    for dataset in LOOKUP.keys():
        lag = LOOKUP[dataset][0]
        if len(LOOKUP[dataset]) > 1:
            external_forecast_horizon = LOOKUP[dataset][1]
        if len(LOOKUP[dataset]) > 2:
            integer_conversion = LOOKUP[dataset][2]
        full_path = os.path.join(datadir, dataset + '.tsf')
        try:
            ds, freq, seasonality, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(full_path, ds_sample_seed=-1, ext_fc_horizon=external_forecast_horizon)
            lengths = [len(ser) for ser in ds['series_value']]
            ds_stats.append(f'{dataset:<40} {ds.shape[0]*np.mean(lengths):<3} values - {str(ds.shape[0]):<3} x {np.mean(lengths)} series (max length {np.max(lengths)})')
            ds_n_vals.append(ds.shape[0]*np.mean(lengths))
        except Exception as e:
            print('ERROR', dataset, e)

    for idx in np.argsort(ds_n_vals):
        print(ds_stats[idx])
