import os
import argparse
import json
import shutil
import tarfile

from monitoring import aggregate_log
from util import basename, PatchedJSONEncoder


def read_json(filepath):
    with open(filepath, 'r') as logf:
        return json.load(logf)


def read_txt(filepath):
    with open(filepath, 'r') as reqf:
        return [line.strip() for line in reqf.readlines()]


def read_monitoring(filepath):
    return aggregate_log(filepath)


def aggregate_results(directory):
    reader_methods = {name: func for name, func in globals().items() if name.startswith('read_')}
    res = {'directory_name': basename(directory)}
    # read all files
    for filename in os.listdir(directory):
        fbase, ext = os.path.splitext(filename)
        reader_method = f'read_{ext[1:]}'
        if reader_method in reader_methods:
            res[fbase] = reader_methods[reader_method](os.path.join(directory, filename))
    return res


def process_directory(directory, output_log_dir=None, output_agglog_dir=None):
    print('Processing', directory)
    # create summary
    if output_agglog_dir is not None: 
        if not os.path.isdir(output_agglog_dir):
            os.makedirs(output_agglog_dir)
        agglog_name = os.path.join(output_agglog_dir, basename(directory) + '.json')
        # load if already exists
        if os.path.isfile(agglog_name):
            with open(agglog_name, 'r') as agglog:
                res = json.load(agglog)
        else:
            res = aggregate_results(directory)
            if output_log_dir is not None:
                res['full_log'] = os.path.join(output_log_dir, basename(directory) + '.tar.gz')
            with open(agglog_name, 'w') as agglog:
                json.dump(res, agglog, indent=4, cls=PatchedJSONEncoder)
    else:
        res = aggregate_results(directory)
    # create tar
    if output_log_dir is not None:
        if not os.path.isdir(output_log_dir):
            os.makedirs(output_log_dir)
        log_tar_name = os.path.join(output_log_dir, basename(directory) + '.tar.gz')
        if not os.path.exists(log_tar_name):
            with tarfile.open(log_tar_name, 'w:gz') as tar:
                for fname in os.listdir(directory):
                    tar.add(os.path.join(directory, fname))
    return res


def main(directory, output_log_dir=None, output_agglog_dir=None, clean=False):
    if clean: # remove all subdirectory contents
        for rootdir in [output_log_dir, output_agglog_dir]:
            if rootdir is not None and os.path.isdir(rootdir):
                for subdir in os.listdir(rootdir):
                    if os.path.isfile(os.path.join(rootdir, subdir)):
                        os.remove(os.path.join(rootdir, subdir))
                    else:
                        shutil.rmtree(os.path.join(rootdir, subdir))
    # process
    results = {}
    for dir in sorted(os.listdir(directory)):
        results[dir] = process_directory(os.path.join(directory, dir), output_log_dir, output_agglog_dir)
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", default="mnt_data/results", type=str, help="directory with experimental result directories")
    parser.add_argument("--output-agglog-dir", default="mnt_data/results_merged", type=str, help="directory where experiments log aggregates (json format) are created")
    parser.add_argument("--output-log-dir", default=None, type=str, help="directory where the full logs shall be stored (.tar.gz archives)")
    parser.add_argument("--clean", action='store_true', help="set to first delete all content in given output directories")

    args = parser.parse_args()
    results = main(args.directory, args.output_log_dir, args.output_agglog_dir, args.clean)
    print('Processed', len(results), 'directories!')
