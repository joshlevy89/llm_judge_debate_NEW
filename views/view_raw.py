"""
Print raw JSON object from a JSONL file. Works for verdicts, debates, and qa.
"""
import argparse
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
results_dir = project_root / 'results'

SEARCH_DIRS = ['verdicts', 'debates', 'qa', 'debate_checks']


def find_file_and_record(run_id, record_id=None, file_type=None):
    dirs = [file_type] if file_type else SEARCH_DIRS
    
    for dir_name in dirs:
        # QA is a single file - search by run_id inside it
        if dir_name == 'qa':
            path = results_dir / 'qa' / 'qa_results.jsonl'
            if path.exists():
                for line in open(path):
                    data = json.loads(line)
                    if data.get('run_id') == run_id:
                        if record_id is None or data.get('record_id') == record_id:
                            return data
        else:
            path = results_dir / dir_name / f'{run_id}.jsonl'
            if path.exists():
                for line in open(path):
                    data = json.loads(line)
                    if record_id is None or data.get('record_id') == record_id:
                        return data
    return None


def main():
    parser = argparse.ArgumentParser(description='Print raw JSON from result files')
    parser.add_argument('run_id')
    parser.add_argument('record_id', nargs='?')
    parser.add_argument('--type', choices=SEARCH_DIRS, help='Specify file type instead of auto-detecting')
    args = parser.parse_args()
    
    record = find_file_and_record(args.run_id, args.record_id, args.type)
    if not record:
        print(f"Record not found for run_id={args.run_id}" + (f", record_id={args.record_id}" if args.record_id else ""))
        return
    
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
