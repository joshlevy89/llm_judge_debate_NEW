"""
Prints the QA results for a given run_id and/or record_id.
Also prints the raw model response and the parsed model response.
"""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='View QA results')
    parser.add_argument('run_id')
    parser.add_argument('record_id')
    args = parser.parse_args()
    
    path = Path('results') / 'qa' / 'qa_results.jsonl'
    
    for line in open(path):
        data = json.loads(line)
        if args.run_id and data.get('run_id') != args.run_id:
            continue
        if args.record_id and data.get('record_id') != args.record_id:
            continue
        
        if not data.get('success', True):
            print(f"{'='*80}")
            print(f"ERROR: QA failed")
            print(f"{'='*80}")
            print(f"Run ID: {data.get('run_id')}")
            print(f"Record ID: {data.get('record_id')}")
            print(f"Question Idx: {data.get('question_idx')}")
            print(f"Error message: {data.get('error_message', 'Unknown error')}")
            if args.record_id:
                break
            continue
        
        print(f"{'='*80}")
        print(f"Run: {data['run_id']} | Record: {data['record_id']} | Question Idx: {data['question_idx']}")
        print(f"{'='*80}\nQuestion\n{'='*80}")
        print(f"{data['question']}\n")
        print(f"Options: {data['options']}")
        print(f"{'='*80}\nRaw Model Response\n{'='*80}")
        print(f"{data['raw_model_response']}")
        print(f"{'='*80}\nParsed Model Response\n{'='*80}")
        parsed_response = data['parsed_model_response']
        print(f"Answer: {parsed_response['answer']}")
        print(f"Confidence: {parsed_response['confidence']}")
        print(f"Reasoning: {parsed_response['reasoning']}")
        print(f"{'='*80}\nOutput\n{'='*80}")
        print(f"Correct Answer: {data['correct_idx']}")
        print(f"Model Answer: {parsed_response['answer']}")
        print(f"Correct: {parsed_response['answer'] == data['correct_idx']}")

        if args.record_id:
            break

if __name__ == '__main__':
    main()

