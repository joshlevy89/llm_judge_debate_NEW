import json
import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from datetime import datetime as dt

def format_datetime(datetime_str):
    if not datetime_str:
        return ''
    try:
        d = dt.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return d.strftime('%Y-%m-%d %H:%M')
    except:
        return datetime_str[:16] if len(datetime_str) > 16 else datetime_str

def shorten_name(name):
    if not name or not isinstance(name, str):
        return name
    parts = name.split('/')
    return parts[-1] if len(parts) > 1 else name

def read_first_record(file_path):
    with open(file_path) as f:
        return json.loads(f.readline())

def count_records(file_path):
    with open(file_path) as f:
        return sum(1 for line in f if line.strip())

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
        
        record = read_first_record(file_path)
        row = {
            'debate_id': file_path.stem,
            'n_records': n_records,
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
            'datetime': format_datetime(record.get('datetime', ''))
        }
        row.update(flatten_config(debate_config, prefix='d_'))
        row.update(flatten_config({k: v for k, v in verdict_config.items() if k != 'debate_run_id'}, prefix='v_'))
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('datetime').drop(columns=['datetime'])
    return df

def main():
    parser = argparse.ArgumentParser(description='Display experimental parameters')
    parser.add_argument('type', choices=['verdicts', 'debates'], 
                       help='Type of experiments to display')
    args = parser.parse_args()
    
    if args.type == 'verdicts':
        df = get_verdict_configs()
        index_col = 'verdict_id'
    else:
        df = get_debate_configs()
        index_col = 'debate_id'
    
    if df.empty:
        print("No data found")
        return
    
    df = df.dropna(axis=1, how='all')
    df = df.set_index(index_col)
    
    print(tabulate(df, headers='keys', tablefmt='simple', showindex=True))

if __name__ == '__main__':
    main()

