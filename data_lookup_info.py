LOOKUP = { # taken from https://github.com/rakshitha123/TSForecasting
    "cif_2016_dataset": ( 15, 6),
    "nn5_daily_dataset_without_missing_values": ( 9, ),
    "tourism_yearly_dataset": ( 2, ),
    "tourism_quarterly_dataset": ( 5, ),
    "tourism_monthly_dataset": ( 15, ),
    "m1_yearly_dataset": ( 2, ),
    "m1_quarterly_dataset": ( 5, ),
    "m1_monthly_dataset": ( 15, ),
    "m3_yearly_dataset": ( 2, ),
    "m3_quarterly_dataset": ( 5, ),
    "m3_monthly_dataset": ( 15, ),
    "m3_other_dataset": ( 2, ),
    "m4_quarterly_dataset": ( 5, ),
    "m4_monthly_dataset": ( 15, ),
    "m4_weekly_dataset": ( 65, ),
    "m4_daily_dataset": ( 9, ),
    "m4_hourly_dataset": ( 210, ),
    "car_parts_dataset_without_missing_values": ( 15, 12, True),
    "hospital_dataset": ( 15 ,12, True),
    "fred_md_dataset": ( 15, 12),
    "nn5_weekly_dataset": ( 65, 8),
    "traffic_weekly_dataset": ( 65, 8),
    "electricity_weekly_dataset": ( 65, 8, True),
    "solar_weekly_dataset": ( 6, 5),
    "kaggle_web_traffic_weekly_dataset": ( 10, 8, True),
    "dominick_dataset": ( 10, 8),
    "us_births_dataset": ( 9, 30, True),
    "saugeenday_dataset": ( 9, 30),
    "sunspot_dataset_without_missing_values": ( 9, 30, True),
    "covid_deaths_dataset": ( 9, 30, True),
    "weather_dataset": ( 9, 30),
    "traffic_hourly_dataset": ( 30, 168),
    "electricity_hourly_dataset": ( 30, 168, True),
    "solar_10_minutes_dataset": ( 50, 1008),
    "kdd_cup_2018_dataset_without_missing_values": ( 210, 168),
    "pedestrian_counts_dataset": ( 210, 24, True),
    "bitcoin_dataset_without_missing_values": ( 9, 30),
    "vehicle_trips_dataset_without_missing_values": ( 9, 30, True),
    "australian_electricity_demand_dataset": ( 420, 336),
    "rideshare_dataset_without_missing_values": ( 210, 168),
    "temperature_rain_dataset_without_missing_values": ( 9, 30)
}

def set_temporal_budget():
    import json
    import pandas as pd
    from strep.util import read_json

    ds_meta = 'meta_dataset.json'
    dataset_meta = read_json(ds_meta)
    database = pd.read_pickle('results/logs.pkl')

    for ds in dataset_meta.keys():
        ds_results = database[database['dataset'] == ds]
        train_times = ds_results['train_running_time'].dropna()
        if train_times.size > 0:
            dataset_meta[ds]['budget'] = int(train_times.max())
        else: # dataset not processed yet
            dataset_meta[ds]['budget'] = int(database['train_running_time'].dropna().max())

    with open(ds_meta, 'w') as meta:
        json.dump(dataset_meta, meta, indent=4)

if __name__ == "__main__":
    set_temporal_budget()
