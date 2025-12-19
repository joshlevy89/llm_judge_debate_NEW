"""
This is a thin wrapper around view_verdict.py that allows us to view multiple verdicts with one command.
Each verdict should be clearly separate with multiple new lines. 
The script can accept either a run id and a list of verdict record ids OR
it can accept just a run id and it will randomly sample N record ids to show
"""
import argparse
import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from views.view_verdict import display_full_verdict
from analysis.analysis_utils import prepare_df


def get_all_record_ids(verdict_run_id):
    verdict_path = project_root / 'results' / 'verdicts' / f'{verdict_run_id}.jsonl'
    record_ids = []
    with open(verdict_path) as f:
        for line in f:
            data = json.loads(line)
            record_ids.append(data['record_id'])
    return record_ids


def main():
    parser = argparse.ArgumentParser(description='View multiple verdict results')
    parser.add_argument('verdict_run_id', help='Verdict run ID')
    # parser.add_argument('record_ids', nargs='*', help='Record IDs to display (optional)')
    parser.add_argument('--sample', '-n', type=int, help='Randomly sample N records if no record_ids provided')
    # parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling (default: 42)')
    parser.add_argument('--judge-qa-result', type=lambda x: x.lower() == 'true', help='Filter by judge QA correctness (True/False)')
    parser.add_argument('--verdict-result', type=lambda x: x.lower() == 'true', help='Filter by verdict correctness (True/False)')
    parser.add_argument('--hide-private', action='store_true', help='Hide private reasoning')
    parser.add_argument('--view-qa', action='store_true', help='View the associated judge QA')
    args = parser.parse_args()
    
    df = prepare_df(['verdicts', 'debates', 'qa'], specific_verdict_ids=[args.verdict_run_id])

    print(df.shape)
    if args.verdict_result is not None:
        df = df[df['is_correct_verdict'] == args.verdict_result]
    if args.judge_qa_result is not None:
        df = df[df['is_correct_qa_judge'] == args.judge_qa_result]
    
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)))
    record_ids = df['record_id_verdicts']

    print(f'Matched {df.shape[0]} records...')
    
    for i, record_id in enumerate(record_ids):
        if i > 0:
            print('\n\n\n')
        print(f'Transcript For Debate {i}\n')
        
        display_full_verdict(args.verdict_run_id, record_id, 
                           hide_private=args.hide_private,
                           view_qa=args.view_qa)


if __name__ == '__main__':
    main()