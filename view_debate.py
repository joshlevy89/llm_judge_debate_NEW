"""
Prints the debate history for a given run_id and record_id.
Also prints the raw judge response and the parsed judge response.
"""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='View debate results')
    parser.add_argument('run_id', help='Run ID to view')
    parser.add_argument('record_id', nargs='?', help='Optional record ID to filter')
    args = parser.parse_args()
    
    path = Path('results') / 'debate' / f'{args.run_id}.jsonl'
    
    for line in open(path):
        data = json.loads(line)
        if args.record_id and data['record_id'] != args.record_id:
            continue
        
        print(f"{'='*80}")
        print(f"Run: {data['run_id']} | Record: {data['record_id']}")
        print(f"{'='*80}")
        print(f"Question: {data['question']}\n")
        print(f"Options: {data['options']}")
        print(f"{'='*80}")
        print(f"Debate:{data['debate_history_text']}")
        print(f"{'='*80}")
        print(f"Judge Raw Response:\n{data['judge_verdict']['raw_response']}")
        print(f"{'='*80}")
        print(f"Judge Parsed Response:")
        parsed_response = data['judge_verdict']['parsed']
        print(f"Answer: {parsed_response['answer']}")
        print(f"Confidence: {parsed_response['confidence']}")
        print(f"Reasoning: {parsed_response['reasoning']}")
        print(f"{'='*80}")
        print(f"Correct Answer: {data['correct_idx']}")
        print(f"Judge Answer: {parsed_response['answer']}")
        print(f"Correct: {parsed_response['answer'] == data['correct_idx']}")

        if args.record_id:
            break

if __name__ == '__main__':
    main()