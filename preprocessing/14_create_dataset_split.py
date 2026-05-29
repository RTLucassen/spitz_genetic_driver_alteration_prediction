"""
Create dataset split.
"""

import random

import pandas as pd
from pathlib import Path


def select_shortest_list(d):
    if not d:
        raise ValueError("Dictionary is empty")

    lengths = {k: len(v) for k, v in d.items()}
    min_len = min(lengths.values())
    shortest_keys = [k for k, l in lengths.items() if l == min_len]

    key = random.choice(shortest_keys)
    return key

# define path
path = Path(r"")
output_path = Path(r"")

# define settings
test_fraction = 0.25
N_folds = 5
seed = 1


if __name__ == '__main__':

    random.seed(seed)

    df = pd.read_excel(path)
    df['split'] = None

    for center in df['center'].unique():

        df_center = df[df['center'] == center]

        train_val_patients = []
        test_patients = []

        # stratified test split per label
        for label in df_center['label'].unique():

            df_label = df_center[df_center['label'] == label]
            patient_indices = df_label['patient_index'].unique().tolist()
            random.shuffle(patient_indices)

            n_test = int(round(len(patient_indices) * test_fraction))
            test_patients.extend(patient_indices[:n_test])
            train_val_patients.extend(patient_indices[n_test:])

        # assign folds stratified by label within train_val
        split = {i: [] for i in range(N_folds)}

        df_train_val = df_center[df_center['patient_index'].isin(train_val_patients)]

        for label in df_train_val['label'].unique():

            df_label = df_train_val[df_train_val['label'] == label]
            patient_indices = df_label['patient_index'].unique().tolist()
            random.shuffle(patient_indices)

            for i, idx in enumerate(patient_indices):
                fold_id = i % N_folds
                split[fold_id].append(idx)

        # assign to dataframe
        for fold_id, patient_list in split.items():
            df.loc[
                (df['center'] == center) &
                (df['patient_index'].isin(patient_list)),
                'split'
            ] = fold_id

        df.loc[
            (df['center'] == center) &
            (df['patient_index'].isin(test_patients)),
            'split'
        ] = 'test'

    df.to_excel(output_path, index=False)