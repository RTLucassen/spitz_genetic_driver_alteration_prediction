"""
Print dataset characteristics for each split
"""

import pandas as pd
from pathlib import Path

split_path = Path(r"")
split_col_name = ''

if __name__ == '__main__':

    df = pd.read_excel(split_path)
    for center in ['UMCU', 'CLB']:
        df_center = df[df['center'] == center]
        print(center)
        for split in [0,1,2,3,4,'test']:
            df_split = df_center[df_center[split_col_name] == split]
            print(f'split {split}')
            for label in ['ALK', 'ROS1', 'NTRK', 'BRAF', 'HRAS', 'MAP3K8']:
                df_label = df_split[df_split['label'] == label]
                print(f'{label}: {len(df_label)} slides, {len(set(df_label["specimen_index"].tolist()))} specimens')
            print('')
            