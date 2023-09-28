#!/bin/bash

for d in "cif_2016_dataset" "m1_quarterly_dataset" "fred_md_dataset" "tourism_quarterly_dataset" "solar_weekly_dataset" "electricity_weekly_dataset" "nn5_daily_dataset_without_missing_values" "tourism_monthly_dataset" "m1_monthly_dataset" "nn5_weekly_dataset" "hospital_dataset" "m3_quarterly_dataset" "car_parts_dataset_without_missing_values" "m4_weekly_dataset" "traffic_weekly_dataset" "m3_monthly_dataset" "m4_hourly_dataset" "dominick_dataset" "australian_electricity_demand_dataset"
do
    for m in "deepar" "deepstate" "deepfactor" "deeprenewal" "gpforecaster" "mqcnn" "mqrnn" "nbeats" "tempfus" "wavenet" "feedforward"
    do
        for s in "42" "135" "468" "129" "124" "-1"
        do
        python run.py --dataset $d --model $m --ds-seed $s --output-dir $1
        done
    done

    python run.py --dataset $d --model autokeras --ds-seed -1 --output-dir $2
done
# TODO errors for "m1_yearly_dataset" "tourism_yearly_dataset" "m3_other_dataset" "m3_yearly_dataset"
# TODO if there is enough time, also run on ""m4_quarterly_dataset"?
# TODO fix parameter assessment error for "naiveseasonal" "rotbaum"
# TODO fix error for "transformer"
# TODO also run "arima" (takes very long)
# TODO only single series -> no subsampling possible! "us_births_dataset" "saugeenday_dataset" "sunspot_dataset_without_missing_values"
