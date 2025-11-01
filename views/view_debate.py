"""
Prints the debate history for a given run_id and record_id.
Optionally shows verdict from a separate verdict run.
"""
import argparse
import json
from pathlib import Path
from utils.debate_utils import format_debate_history

def main():
    parser = argparse.ArgumentParser(description='View debate results')
    parser.add_argument('run_id')
    parser.add_argument('record_id')
    parser.add_argument('--hide-private', action='store_true', help='Hide private reasoning')
    parser.add_argument('--verdict', help='Verdict run ID to show judgment from')
    args = parser.parse_args()
    
    debate_path = Path('results') / 'debates' / f'{args.run_id}.jsonl'
    
    debate_data = None
    for line in open(debate_path):
        data = json.loads(line)
        if data['record_id'] == args.record_id:
            debate_data = data
            break
    
    if not debate_data:
        print(f"Record {args.record_id} not found in debate run {args.run_id}")
        return
    
    if args.hide_private:
        show_private = False
    else:
        show_private = debate_data['config'].get('private_scratchpad', False)
    debate_text = format_debate_history(debate_data['debate_history'], show_private=show_private)
    
    print(f"{'='*80}")
    print(f"Debate Run: {debate_data['run_id']} | Record: {debate_data['record_id']}")
    print(f"{'='*80}\nQuestion\n{'='*80}")
    print(f"{debate_data['question']}\n")
    print(f"Options: {debate_data['options']}")
    print(f"{'='*80}\nDebate\n{'='*80}")
    print(f"{debate_text}")
    
    if args.verdict:
        verdict_path = Path('results') / 'verdicts' / f'{args.verdict}.jsonl'
        if not verdict_path.exists():
            print(f"\nVerdict run {args.verdict} not found")
            return
        
        verdict_data = None
        for line in open(verdict_path):
            data = json.loads(line)
            if data['record_id'] == args.record_id:
                verdict_data = data
                break
        
        if not verdict_data:
            print(f"\nRecord {args.record_id} not found in verdict run {args.verdict}")
            return
        
        print(f"{'='*80}\nJudge (Verdict Run: {args.verdict})\n{'='*80}")
        if verdict_data['judge_verdict'].get('internal_model_reasoning') is not None:
            print(f"[BEGIN INTERNAL REASONING]\n{verdict_data['judge_verdict']['internal_model_reasoning']}\n[END INTERNAL REASONING]\n")
        print(f"[BEGIN RAW RESPONSE]\n{verdict_data['judge_verdict']['raw_response']}\n[END RAW RESPONSE]")
        print(f"{'='*80}\nJudge Parsed Response\n{'='*80}")
        parsed_response = verdict_data['judge_verdict']['parsed']
        print(f"Answer: {parsed_response['answer']}")
        print(f"Confidence: {parsed_response['confidence']}")
        print(f"Reasoning: {parsed_response['reasoning']}")
        print(f"{'='*80}\nOutput\n{'='*80}")
        print(f"Correct Answer: {verdict_data['correct_idx']}")
        print(f"Judge Verdict: {parsed_response['answer']}")
        print(f"Correct: {parsed_response['answer'] == verdict_data['correct_idx']}")
    else:
        print(f"{'='*80}\nNo Verdict\n{'='*80}")
        print("Use --verdict <verdict_run_id> to view a judgment")

if __name__ == '__main__':
    main()