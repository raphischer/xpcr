import argparse
from datetime import timedelta
import json
import os
import pickle
import time
import sys
import re
import traceback

from methods import init_model_and_data, evaluate
from util import fix_seed, create_output_dir, Logger, PatchedJSONEncoder
from monitoring import Monitoring, monitor_flops_papi


def main(args):
    print(f'Running evaluation on {args.dataset} for {args.model}')
    t0 = time.time()
    args.seed = fix_seed(args.seed)

    ############## TRAINING ##############
    output_dir = create_output_dir(args.output_dir, 'train', args.__dict__)

    try:

        lookup = {
            "cif_6_dataset": ( 15, 6),
            "cif_12_dataset": ( 15, 12),
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
        setattr(args, 'lag', lookup[args.dataset][0])
        if len(lookup[args.dataset]) > 1:
            setattr(args, 'external_forecast_horizon', lookup[args.dataset][1])
        if len(lookup[args.dataset]) > 2:
            setattr(args, 'integer_conversion', lookup[args.dataset][2])

        # tmp = sys.stdout # reroute the stdout to logfile, remember to call close!
        # sys.stdout = Logger(os.path.join(output_dir, f'logfile.txt')),

        ts_train, ts_test, estimator = init_model_and_data(args)

        monitoring = Monitoring(0, args.cpu_monitor_interval, output_dir)
        start_time = time.time()
        predictor = estimator.train(training_data=ts_train)
        end_time = time.time()
        monitoring.stop()

        results = {
            'history': {}, # TODO track history
            'start': start_time,
            'end': end_time,
            'model': None
        }
        # write results
        with open(os.path.join(output_dir, f'results.json'), 'w') as rf:
            json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)



        split = 'validation' ############## INFERENCE ##############
        setattr(args, 'train_logdir', output_dir)
        output_dir = create_output_dir(args.output_dir, 'infer', args.__dict__)
        monitoring = Monitoring(0, args.cpu_monitor_interval, output_dir, split)
        start_time = time.time()

        metrics = evaluate(predictor, ts_test)

        end_time = time.time()
        monitoring.stop()

        results = {
            'metrics': metrics,
            'start': start_time,
            'end': end_time,
            'model': None,
            'data': None
        }

        # write results
        with open(os.path.join(output_dir, f'{split}_results.json'), 'w') as rf:
            json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)



        ############## FNALIZE ##############

        print(f"Evaluation finished in {timedelta(seconds=int(time.time() - t0))} seconds, results can be found in {output_dir}\n")
        # sys.stdout.close(),
        # sys.stdout = tmp
        return output_dir

    except Exception as e:
        with open(os.path.join(output_dir, f'error.txt'), 'a') as f:
            f.write(str(e)),
            f.write(traceback.format_exc()),


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    default='nn5_daily_dataset_without_missing_values')
    parser.add_argument('--model',      default='feed_forward')
    parser.add_argument('--output-dir', default='mnt_data/results')
    parser.add_argument('--epochs',     default=100)
    parser.add_argument('--datadir',    default='mnt_data/data')

    # randomization and hardware profiling
    parser.add_argument("--cpu-monitor-interval", default=.01, type=float, help="Setting to > 0 activates CPU profiling every X seconds")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use (if -1, uses and logs random seed)")

    args = parser.parse_args()

    main(args)
