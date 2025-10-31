"""
Prints the debate history for a given run_id and record_id.
Also prints the raw judge response and the parsed judge response.
"""
import argparse
import json
from pathlib import Path
from debate_utils import format_debate_history

def main():
    parser = argparse.ArgumentParser(description='View debate results')
    parser.add_argument('run_id')
    parser.add_argument('record_id')
    # add an optional argument where can force to hide private reasoning
    parser.add_argument('--hide-private', action='store_true', help='Hide private reasoning')
    args = parser.parse_args()
    
    path = Path('results') / 'debate' / f'{args.run_id}.jsonl'
    
    for line in open(path):
        data = json.loads(line)
        if args.record_id and data['record_id'] != args.record_id:
            continue
        
        if args.hide_private:
            show_private = False
        else:
            show_private = data['config'].get('private_scratchpad', False)
        debate_text = format_debate_history(data['debate_history'], show_private=show_private)
        
        print(f"{'='*80}")
        print(f"Run: {data['run_id']} | Record: {data['record_id']}")
        print(f"{'='*80}\nQuestion\n{'='*80}")
        print(f"{data['question']}\n")
        print(f"Options: {data['options']}")
        print(f"{'='*80}\nDebate\n{'='*80}")
        print(f"{debate_text}")
        print(f"{'='*80}\nJudge\n{'='*80}")
        if data['judge_verdict'].get('internal_model_reasoning') is not None:
            print(f"[BEGIN INTERNAL REASONING]\n{data['judge_verdict']['internal_model_reasoning']}\n[END INTERNAL REASONING]\n")
        print(f"[BEGIN RAW RESPONSE]\n{data['judge_verdict']['raw_response']}\n[END RAW RESPONSE]")
        print(f"{'='*80}\nJudge Parsed Response\n{'='*80}")
        parsed_response = data['judge_verdict']['parsed']
        print(f"Answer: {parsed_response['answer']}")
        print(f"Confidence: {parsed_response['confidence']}")
        print(f"Reasoning: {parsed_response['reasoning']}")
        print(f"{'='*80}\nOutput\n{'='*80}")
        print(f"Correct Answer: {data['correct_idx']}")
        print(f"Judge Verdict: {parsed_response['answer']}")
        print(f"Correct: {parsed_response['answer'] == data['correct_idx']}")

        if args.record_id:
            break

if __name__ == '__main__':
    main()