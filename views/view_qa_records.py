import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tabulate import tabulate
from analysis.analysis_utils import load_qa
from views.view_utils import shorten_name

IGNORED_COLS = ['config_num_questions', 'config_max_threads', 'config_specific_question_idxs', 'config_rerun']

def get_groupable_cols(df):
    config_cols = [col for col in df.columns if col.startswith('config_') and col not in IGNORED_COLS]
    
    groupable = []
    for col in config_cols:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            continue
        groupable.append(col)
    
    return groupable

def prepare_data(df, groupable_cols):
    if 'config_random_seed' in df.columns:
        df['config_random_seed'] = df['config_random_seed'].fillna(42)
    
    for col in groupable_cols:
        if 'model' in col or 'dataset' in col:
            df[col] = df[col].apply(shorten_name)
    
    return df

def aggregate_by_config(df, groupable_cols):
    grouped = df.groupby(groupable_cols, dropna=False).agg(
        n_records=('is_correct', 'count'),
        n_correct=('is_correct', 'sum'),
        accuracy=('is_correct', 'mean')
    ).reset_index()
    
    grouped['accuracy'] = grouped['accuracy'].round(3)
    return grouped.sort_values(groupable_cols)

def main():
    df = load_qa()
    
    groupable_cols = get_groupable_cols(df)
    if not groupable_cols:
        print("No groupable config columns found")
        return
    
    df = prepare_data(df, groupable_cols)
    grouped = aggregate_by_config(df, groupable_cols)
    
    display_cols = groupable_cols + ['n_records', 'n_correct', 'accuracy']
    print(tabulate(grouped[display_cols], headers='keys', tablefmt='simple', showindex=False))

if __name__ == '__main__':
    main()