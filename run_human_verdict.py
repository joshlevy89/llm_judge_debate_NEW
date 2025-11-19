import argparse
from views.view_debate import display_debate, load_debate_data
from pathlib import Path
import json

def setup_output_path():
    output_dir = Path('results') / 'human'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'human_verdicts.jsonl'

def main():
    parser = argparse.ArgumentParser(description='View debate results')
    parser.add_argument('run_id')
    parser.add_argument('record_id', nargs='?', help='Optional record ID to view specific record')
    parser.add_argument('--hide-private', action='store_true', help='Hide private reasoning')
    parser.add_argument('--random_id', action='store_true', help='Pick a random record from the file')
    args = parser.parse_args()
    
    debate_data = load_debate_data(args.run_id, record_id=args.record_id, random_id=args.random_id)
    if not debate_data:
        print(f"Record {args.record_id} not found in debate run {args.run_id}")
        return
    
    display_debate(debate_data, hide_private=args.hide_private, do_latex_formatting=True)

    results_path = setup_output_path()

    print("="*80)

    print("Verdict (index): ")
    verdict = input("> ").strip()
    
    print("Confidence (0-100): ")
    confidence = input("> ").strip()

    print("Reasoning: ")
    reasoning = input("> ").strip()

    print(f"Correct Idx {debate_data['correct_idx']}")

    result = {
        'record_id': debate_data['record_id'],
        'verdict': verdict,
        'confidence': confidence,
        'reasoning': reasoning
    }

    with open(results_path, 'a') as f:
        f.write(json.dumps(result) + '\n')
        f.flush()

if __name__ == "__main__":
    main()