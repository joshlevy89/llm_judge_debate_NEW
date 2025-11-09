import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tabulate import tabulate
from analysis.analysis_utils import prepare_df
from views.view_utils import shorten_name, apply_filters, get_varying_cols

IGNORED_COLS = ['config_num_questions', 'config_max_threads', 'config_specific_question_idxs', 'config_rerun']

def get_groupable_cols(df):
    config_cols = [col for col in df.columns if col.startswith('config_')]
    
    groupable = []
    for col in config_cols:
        if any(ignored in col for ignored in IGNORED_COLS):
            continue
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
        n_records=('is_correct_qa', 'count'),
        n_correct=('is_correct_qa', 'sum'),
        accuracy=('is_correct_qa', 'mean')
    ).reset_index()
    
    grouped['accuracy'] = grouped['accuracy'].round(3)
    return grouped.sort_values(groupable_cols)

def main():
    parser = argparse.ArgumentParser(description='Display QA accuracy by configuration')
    parser.add_argument('--filter', nargs='+', metavar='field=value',
                       help='Filter by field values (applied before aggregation)')
    parser.add_argument('--varying', action='store_true',
                       help='Show only columns with varying values')
    args = parser.parse_args()
    
    df = prepare_df(types=['qa'])
    if df.empty:
        print("No data found")
        return
    
    groupable_cols = get_groupable_cols(df)
    if not groupable_cols:
        print("No groupable config columns found")
        return
    
    df = prepare_data(df, groupable_cols)
    
    df = apply_filters(df, args.filter)
    if df is None:
        return
    
    grouped = aggregate_by_config(df, groupable_cols)
    
    df = grouped.dropna(axis=1, how='all')
    
    metric_cols = ['n_records', 'n_correct', 'accuracy']
    if args.varying:
        varying_cols = get_varying_cols(df, exclude_cols=metric_cols)
        display_cols = varying_cols + metric_cols
    else:
        available_groupable = [col for col in groupable_cols if col in df.columns]
        display_cols = available_groupable + metric_cols
    
    print(tabulate(df[display_cols], headers='keys', tablefmt='simple', showindex=False))

if __name__ == '__main__':
    main()