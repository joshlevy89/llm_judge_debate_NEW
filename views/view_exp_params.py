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

def count_records(file_path):
    with open(file_path) as f:
        return sum(1 for line in f if line.strip())

def get_debate_config(debate_run_id):
    debate_file = Path(f"results/debates/{debate_run_id}.jsonl")
    if not debate_file.exists():
        return None
    
    with open(debate_file) as f:
        first_line = f.readline()
        record = json.loads(first_line)
        return record['config']

def get_verdict_configs():
    verdict_dir = Path("results/verdicts")
    configs = []
    
    for verdict_file in verdict_dir.glob("*.jsonl"):
        verdict_run_id = verdict_file.stem
        n_records = count_records(verdict_file)
        
        if n_records == 0:
            continue
        
        with open(verdict_file) as f:
            first_line = f.readline()
            record = json.loads(first_line)
            
            verdict_config = record['config']
            debate_run_id = verdict_config['debate_run_id']
            debate_config = get_debate_config(debate_run_id)
            
            if debate_config is None:
                continue
            
            row = {
                'verdict_id': verdict_run_id,
                'debate_id': debate_run_id,
                'n_records': n_records,
                'datetime': format_datetime(record.get('datetime', ''))
            }
            
            for key, value in debate_config.items():
                if key in ['debater_model', 'judge_model', 'dataset_name']:
                    value = shorten_name(value)
                row[f'd_{key}'] = value
            
            for key, value in verdict_config.items():
                if key != 'debate_run_id':
                    if key == 'judge_model':
                        value = shorten_name(value)
                    row[f'v_{key}'] = value
            
            configs.append(row)
    
    df = pd.DataFrame(configs)
    if 'datetime' in df.columns:
        if not df.empty:
            df = df.sort_values('datetime')
        df.drop(columns=['datetime'], inplace=True)
    return df

def get_debate_configs():
    debate_dir = Path("results/debates")
    configs = []
    
    for debate_file in debate_dir.glob("*.jsonl"):
        debate_run_id = debate_file.stem
        n_records = count_records(debate_file)
        
        if n_records == 0:
            continue
        
        with open(debate_file) as f:
            first_line = f.readline()
            record = json.loads(first_line)
            config = record['config']
            
            row = {'debate_id': debate_run_id, 'n_records': n_records, 'datetime': format_datetime(record.get('datetime', ''))}
            for key, value in config.items():
                if key in ['debater_model', 'judge_model', 'dataset_name']:
                    value = shorten_name(value)
                row[key] = value
            configs.append(row)
    
    df = pd.DataFrame(configs)
    if 'datetime' in df.columns:
        if not df.empty:
            df = df.sort_values('datetime')
        df.drop(columns=['datetime'], inplace=True)
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

