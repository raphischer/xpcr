import argparse
from datetime import timedelta
import json
import os
import time
import traceback
from pathlib import Path
from codecarbon import OfflineEmissionsTracker

from data_lookup_info import LOOKUP
from methods import init_model_and_data, run_validation, evaluate
from mlprops.util import fix_seed, create_output_dir, PatchedJSONEncoder


def main(args):
    print(f'Running evaluation on {args.dataset} for {args.model}')
    t0 = time.time()

    ############## TRAINING ##############
    output_dir = create_output_dir(args.output_dir, 'train', args.__dict__)
    setattr(args, 'train_logdir', output_dir)
    
    try:

        setattr(args, 'lag', LOOKUP[args.dataset][0])
        if len(LOOKUP[args.dataset]) > 1:
            setattr(args, 'external_forecast_horizon', LOOKUP[args.dataset][1])
        if len(LOOKUP[args.dataset]) > 2:
            setattr(args, 'integer_conversion', LOOKUP[args.dataset][2])

        # tmp = sys.stdout # reroute the stdout to logfile, remember to call close!
        # sys.stdout = Logger(os.path.join(output_dir, f'logfile.txt')),

        ts_train, history, ts_test, model = init_model_and_data(args)
        # set global seed only after data loading, because this uses an internal ds subsampling seed!
        args.seed = fix_seed(args.seed)

        if not hasattr(model, 'train'):
            # no global training needed, model is only a predictor
            pass

        else:
            # model is a GluonTS estimator, that upon calling 'train' returns a predictor
            emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.cpu_monitor_interval, log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir=output_dir)
            emissions_tracker.start()
            start_time = time.time()
            model = model.train(training_data=ts_train)
            end_time = time.time()
            emissions_tracker.stop()

            results = {
                'history': {
                    'loss': history.loss_history if history is not None else None
                },
                'start': start_time,
                'end': end_time,
                'model': None
            }
            # write results
            with open(os.path.join(output_dir, f'results.json'), 'w') as rf:
                json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)

        split = 'validation' ############## INFERENCE ##############
        output_dir = create_output_dir(args.output_dir, 'infer', args.__dict__)
        start_time = time.time()

        num_samples = 100
        emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.cpu_monitor_interval, log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir=output_dir)
        emissions_tracker.start()
        forecast, groundtruth = run_validation(model, ts_test, num_samples)
        forecast = list(forecast)
        groundtruth = list(groundtruth)
        emissions_tracker.stop()
        end_time = time.time()

        try:
            try:
                num_params = model.get_param_count()
            except Exception: # GluonTS model
                num_params = sum([val._reduce().size for val in model.network._collect_params_with_prefix().values()])
            try:
                fsize = model.get_fsize(output_dir)
            except Exception: # GluonTS model
                model.serialize(Path(output_dir))
                relevant_files = [os.path.join(output_dir, fname) for fname in ['input_transform.json', 'parameters.json', 'prediction_net-0000.params', 'prediction_net-network.json', 'type.txt', 'version.json']]
                fsize = sum([os.path.getsize(fname) for fname in relevant_files])

            model_stats = { 'params': num_params, 'fsize': fsize }
        except Exception:
            model_stats = {}

        results = {
            'metrics': evaluate(forecast, groundtruth),
            'start': start_time,
            'end': end_time,
            'num_samples': num_samples,
            'model': model_stats
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
        print('ERROR\n', e)
        with open(os.path.join(output_dir, f'error.txt'), 'a') as f:
            f.write(str(e))
            f.write('\n')
            f.write(traceback.format_exc())
        if "emissions_tracker" in locals().keys():
            emissions_tracker.stop()
        raise RuntimeError(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',                            default='bitcoin_dataset_without_missing_values')
    parser.add_argument('--model',                              default='deepar')
    parser.add_argument('--output-dir',                         default='mnt_data/debug')
    parser.add_argument('--ds-seed', type=int,                  default=-1)
    parser.add_argument('--epochs', type=int,                   default=100)
    parser.add_argument('--datadir',                            default='mnt_data/data')

    # randomization and hardware profiling
    parser.add_argument("--cpu-monitor-interval", type=float,   default=0.5, help="Setting to > 0 activates CPU profiling every X seconds")
    parser.add_argument("--seed", type=int,                     default=42, help="Seed to use (if -1, uses and logs random seed)")

    args = parser.parse_args()

    if args.model != 'autosklearn':
        tmpdir = os.path.join(os.path.dirname(args.output_dir), "tmp", os.path.basename(args.output_dir))
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir)
        os.environ['TMPDIR'] = tmpdir
        print('Using tmp dir', tmpdir)

    main(args)
