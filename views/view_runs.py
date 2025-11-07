import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tabulate import tabulate
from analysis.analysis_utils import prepare_df


# Operational/metadata config params to ignore when determining duplicate configs
IGNORE_FOR_DEDUP = [
    'config_max_threads_verdicts',
    'config_max_threads_debates',
    'config_max_threads_qa',
    'config_rerun_verdicts',
    'config_rerun_debates',
    'config_rerun_qa',
    'config_skip_qa_verdicts',
    'config_skip_qa_debates',
    'config_skip_qa_qa',
    'config_specific_record_ids_verdicts',
    'config_specific_record_ids_debates',
    'config_subset_n_verdicts',
    'config_subset_n_debates',
    'config_subset_n_qa',
]


def get_run_id_col(type_name):
    return {
        'verdicts': f'verdict_run_id_{type_name}',
        'debates': f'run_id_{type_name}',
        'qa': f'run_id_{type_name}'
    }[type_name]


def get_run_summary(df, type_name):
    run_id_col = get_run_id_col(type_name)
    df = df.rename(columns={run_id_col: 'run_id'})
    
    success_col = f'success_{type_name}'
    parsed_col = f'parsed_answer_{type_name}'
    
    grouped = df.groupby('run_id')
    config_cols = sorted([col for col in df.columns if col.startswith('config_')])
    
    summary = grouped.first()[config_cols]
    summary['n_records'] = grouped.size()

    if success_col in df.columns:
        summary['n_error'] = grouped[success_col].apply(lambda x: (x == False).sum())

    if parsed_col in df.columns and success_col in df.columns:
        summary['n_null'] = grouped.apply(lambda g: ((g[success_col] == True) & g[parsed_col].isna()).sum(), include_groups=False)
        summary['n_valid'] = grouped.apply(lambda g: ((g[success_col] == True) & g[parsed_col].notna()).sum(), include_groups=False)
    elif parsed_col in df.columns:
        summary['n_null'] = grouped[parsed_col].apply(lambda x: x.isna().sum())
        summary['n_valid'] = grouped[parsed_col].apply(lambda x: x.notna().sum())
    elif success_col in df.columns:
        summary['n_valid'] = grouped[success_col].sum()
    
    count_cols = [c for c in ['n_records', 'n_error', 'n_null', 'n_valid'] if c in summary.columns]
    return summary.reset_index()[['run_id'] + count_cols + config_cols]


def main():
    parser = argparse.ArgumentParser(description='Display experimental parameters')
    parser.add_argument('type', choices=['verdicts', 'debates', 'qa'], 
                       help='Type of experiments to display')
    parser.add_argument('--filter', nargs='+', metavar='field=value',
                       help='Filter by field values')
    parser.add_argument('--varying', action='store_true',
                       help='Show only columns with varying values')
    parser.add_argument('--best-only', action='store_true',
                       help='Keep only the run with most valid records for duplicate configs')
    args = parser.parse_args()
    
    df = prepare_df(args.type, filter_errors=False, filter_nulls=False)
    if df.empty:
        print("No data found")
        return
    
    df = get_run_summary(df, args.type)
    
    if args.filter:
        for filter_expr in args.filter:
            if '=' not in filter_expr:
                print(f"Invalid filter format: {filter_expr}. Expected field=value")
                return
            field, value = filter_expr.split('=', 1)
            
            if field not in df.columns:
                print(f"Unknown field: {field}. Available fields: {', '.join(df.columns)}")
                return
            
            if pd.api.types.is_numeric_dtype(df[field]):
                value = pd.to_numeric(value)
            df = df[df[field] == value]
        
        if df.empty:
            print("No matching records found")
            return
    
    df = df.dropna(axis=1, how='all')
    
    if args.best_only:
        config_cols = [col for col in df.columns if col.startswith('config_')]
        dedup_cols = [col for col in config_cols if col not in IGNORE_FOR_DEDUP]
        
        if 'n_valid' in df.columns:
            df = df.sort_values(['n_valid', 'n_records'], ascending=False)
        else:
            df = df.sort_values('n_records', ascending=False)
        df = df.drop_duplicates(subset=dedup_cols, keep='first')
    
    if args.varying:
        varying_cols = []
        for col in df.columns:
            if col == 'run_id':
                continue
            try:
                if df[col].nunique(dropna=False) > 1:
                    varying_cols.append(col)
            except TypeError:
                pass
        df = df[['run_id'] + varying_cols]
    
    # Sort by config columns (alphabetically) to group similar configs together
    sort_cols = [col for col in df.columns if col.startswith('config_')]
    if sort_cols:
        df = df.sort_values(sort_cols)
    
    df = df.set_index('run_id')
    print(tabulate(df, headers='keys', tablefmt='simple', showindex=True))

    # Also print out the records as a list
    runs = df.index.tolist()
    print(f'\n# Runs: {len(runs)}, Run IDs: {runs}')
    


if __name__ == '__main__':
    main()
