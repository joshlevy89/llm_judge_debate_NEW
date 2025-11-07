import sys
import json
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tabulate import tabulate
from datetime import datetime as dt
from views.view_utils import shorten_name

def format_datetime(datetime_str):
    if not datetime_str:
        return ''
    try:
        d = dt.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return d.strftime('%Y-%m-%d %H:%M')
    except:
        return datetime_str[:16] if len(datetime_str) > 16 else datetime_str

def read_first_record(file_path):
    with open(file_path) as f:
        return json.loads(f.readline())

def count_records(file_path):
    with open(file_path) as f:
        return sum(1 for line in f if line.strip())

def count_success_failure(file_path):
    success = 0
    failure = 0
    with open(file_path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get('success', True):
                    success += 1
                else:
                    failure += 1
    return success, failure

def flatten_config(config, prefix=''):
    row = {}
    for key, value in config.items():
        if key in ['debater_model', 'judge_model', 'dataset_name']:
            value = shorten_name(value)
        row[prefix + key] = value
    
    if prefix.startswith('d_') and not config.get('private_scratchpad', True):
        row[prefix + 'private_reasoning_word_limit'] = None
    
    return row

def get_debate_configs():
    debate_dir = Path("results/debates")
    rows = []
    
    for file_path in debate_dir.glob("*.jsonl"):
        n_records = count_records(file_path)
        if n_records == 0:
            continue
        
        n_success, n_failure = count_success_failure(file_path)
        record = read_first_record(file_path)
        row = {
            'debate_id': file_path.stem,
            'n_records': n_records,
            'n_success': n_success,
            'n_failure': n_failure,
            'datetime': format_datetime(record.get('datetime', ''))
        }
        row.update(flatten_config(record['config']))
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('datetime').drop(columns=['datetime'])
    return df

def get_verdict_configs():
    verdict_dir = Path("results/verdicts")
    rows = []
    
    for file_path in verdict_dir.glob("*.jsonl"):
        n_records = count_records(file_path)
        if n_records == 0:
            continue
        
        n_success, n_failure = count_success_failure(file_path)
        record = read_first_record(file_path)
        verdict_config = record['config']
        debate_run_id = verdict_config['debate_run_id']
        
        debate_file = Path(f"results/debates/{debate_run_id}.jsonl")
        if not debate_file.exists():
            continue
        
        debate_record = read_first_record(debate_file)
        debate_config = debate_record['config']
        
        row = {
            'verdict_id': file_path.stem,
            'debate_id': debate_run_id,
            'n_records': n_records,
            'n_success': n_success,
            'n_failure': n_failure,
            'datetime': format_datetime(record.get('datetime', ''))
        }
        row.update(flatten_config(debate_config, prefix='d_'))
        row.update(flatten_config({k: v for k, v in verdict_config.items() if k != 'debate_run_id'}, prefix='v_'))
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('datetime').drop(columns=['datetime'])
    return df

def get_qa_configs():
    qa_file = Path("results/qa/qa_results.jsonl")
    if not qa_file.exists():
        return pd.DataFrame()
    
    rows = []
    seen_runs = {}
    
    with open(qa_file) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            run_id = record.get('run_id')
            if not run_id or 'config' not in record:
                continue
            
            if run_id not in seen_runs:
                seen_runs[run_id] = {
                    'config': record['config'],
                    'datetime': record.get('datetime', ''),
                    'count': 0,
                    'success': 0,
                    'failure': 0
                }
            seen_runs[run_id]['count'] += 1
            if record.get('success', True):
                seen_runs[run_id]['success'] += 1
            else:
                seen_runs[run_id]['failure'] += 1
    
    for run_id, data in seen_runs.items():
        row = {
            'run_id': run_id,
            'n_records': data['count'],
            'n_success': data['success'],
            'n_failure': data['failure'],
            'datetime': format_datetime(data['datetime'])
        }
        row.update(flatten_config(data['config']))
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # if not df.empty:
        # df = df.sort_values('datetime').drop(columns=['datetime'])
    return df

def main():
    parser = argparse.ArgumentParser(description='Display experimental parameters')
    parser.add_argument('type', choices=['verdicts', 'debates', 'qa'], 
                       help='Type of experiments to display')
    parser.add_argument('--filter', nargs='+', metavar='field=value',
                       help='Filter by field values (e.g., --filter debater_model=gpt-4 n_records=100)')
    parser.add_argument('--varying', action='store_true',
                       help='Show only columns with varying values')
    args = parser.parse_args()
    
    if args.type == 'verdicts':
        df = get_verdict_configs()
        index_col = 'verdict_id'
    elif args.type == 'qa':
        df = get_qa_configs()
        index_col = 'run_id'
    else:
        df = get_debate_configs()
        index_col = 'debate_id'
    
    if df.empty:
        print("No data found")
        return
    
    if args.filter:
        for filter_expr in args.filter:
            if '=' not in filter_expr:
                print(f"Invalid filter format: {filter_expr}. Expected field=value")
                return
            field, value = filter_expr.split('=', 1)
            
            if field not in df.columns:
                print(f"Unknown field: {field}. Available fields: {', '.join(df.columns)}")
                return
            
            try:
                if pd.api.types.is_numeric_dtype(df[field]):
                    value = pd.to_numeric(value)
                df = df[df[field] == value]
            except ValueError:
                df = df[df[field].astype(str) == value]
        
        if df.empty:
            print("No matching records found")
            return
    
    df = df.dropna(axis=1, how='all')
    
    if args.varying:
        varying_cols = [col for col in df.columns if col != index_col and df[col].nunique() > 1]
        df = df[[index_col] + varying_cols]
    
    df = df.set_index(index_col)
    
    print(tabulate(df, headers='keys', tablefmt='simple', showindex=True))

if __name__ == '__main__':
    main()

