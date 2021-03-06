#!/usr/bin/env python
import os
import sys
from time import time
import argparse
import numpy as np
import pandas as pd
from davis2017.evaluation import DAVISEvaluation

# Default params go here.
default_dataset_path = f'./datasets/DAVIS'
default_results_path = f'./results/DAVIS/rvos'

time_start = time()
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', type=str, help='Path to VOS dataset that contains Annotations, JPEGImages etc.',
                    required=False, default=default_dataset_path)

parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    required=False, default=default_results_path)

parser.add_argument('--img_folder', type=str, help='Name of the folder containing the  image sequence folders',
                    required=False, default='JPEGImages')

parser.add_argument('--mask_folder', type=str, help='Path to the folder containing the mask sequence folders',
                    required=False, default='Annotations')

parser.add_argument('--imagesets_path', type=str, help='Path to a specific imageset (txt)', default=None)

args, _ = parser.parse_known_args()


csv_name_global = f'global_results.csv'
csv_name_per_sequence = f'per-sequence_results.csv'


# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(args.results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)



if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences')
    # Create dataset and evaluate

    dataset_eval = DAVISEvaluation(dataset_root=args.dataset_root,
                                   img_folder=args.img_folder,
                                   mask_folder=args.mask_folder,
                                   imagesets_path=args.imagesets_path)

    metrics_res = dataset_eval.evaluate(args.results_path)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')







# Print the results
sys.stdout.write(f"--------------------------- Global results ---------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for  ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
