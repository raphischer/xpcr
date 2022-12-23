#!/bin/bash
for d in "tsfcif_6_dataset" "tsfcif_12_dataset" "tsfnn5_daily_dataset_without_missing_values" "tsftourism_yearly_dataset" "tsftourism_quarterly_dataset" "tsftourism_monthly_dataset" "tsfm1_yearly_dataset" "tsfm1_quarterly_dataset" "tsfm1_monthly_dataset" "tsfm3_yearly_dataset" "tsfm3_quarterly_dataset" "tsfm3_monthly_dataset" "tsfm3_other_dataset" "tsfm4_quarterly_dataset" "tsfm4_monthly_dataset" "tsfm4_weekly_dataset" "tsfm4_daily_dataset" "tsfm4_hourly_dataset" "tsfcar_parts_dataset_without_missing_values" "tsfhospital_dataset" "tsffred_md_dataset" "tsfnn5_weekly_dataset" "tsftraffic_weekly_dataset" "tsfelectricity_weekly_dataset" "tsfsolar_weekly_dataset" "tsfkaggle_web_traffic_weekly_dataset" "tsfdominick_dataset" "tsfus_births_dataset" "tsfsaugeenday_dataset" "tsfsunspot_dataset_without_missing_values" "tsfcovid_deaths_dataset" "tsfweather_dataset" "tsftraffic_hourly_dataset" "tsfelectricity_hourly_dataset" "tsfsolar_10_minutes_dataset" "tsfkdd_cup_2018_dataset_without_missing_values" "tsfpedestrian_counts_dataset" "tsfbitcoin_dataset_without_missing_values" "tsfvehicle_trips_dataset_without_missing_values" "tsfaustralian_electricity_demand_dataset" "tsfrideshare_dataset_without_missing_values" "tsftemperature_rain_dataset_without_missing_values"
do
    for m in "feed_forward" "deepar" "nbeats" "wavenet" "transformer"
    do
        python run.py --dataset $d --model $m
    done
done
