"""
Prints the QA results for a given run_id and/or record_id.
Also prints the raw model response and the parsed model response.
"""
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.analysis_utils import prepare_df

def display_qa(row, display_question=True):
    if not row.get('success_qa', True):
        print(f"{'-'*80}")
        print(f"ERROR: QA failed")
        print(f"{'-'*80}")
        print(f"Run ID: {row.get('run_id_qa')}")
        print(f"Record ID: {row.get('record_id_qa')}")
        print(f"Question Idx: {row.get('question_idx_qa')}")
        print(f"Error message: {row.get('error_message_qa', 'Unknown error')}")
        return

    if display_question:
        print(f"{'='*80}")
        print(f"Run: {row['run_id_qa']} | Record: {row['record_id_qa']} | Question Idx: {row['question_idx_qa']}")
        print(f"{'='*80}\nQuestion\n{'='*80}")
        print(f"{row['question_qa']}\n")
        print(f"Options: {row['options_qa']}")
    print(f"{'-'*80}\nRaw Model Response\n{'-'*80}")
    print(f"{row['raw_model_response_qa']}")
    print(f"{'-'*80}\nParsed Model Response\n{'-'*80}")
    parsed_response = row['parsed_model_response_qa']
    if parsed_response and isinstance(parsed_response, dict):
        print(f"Answer: {parsed_response.get('answer')}")
        print(f"Confidence: {parsed_response.get('confidence')}")
        print(f"Reasoning: {parsed_response.get('reasoning')}")
    print(f"{'-'*80}\nOutput\n{'-'*80}")
    print(f"Correct Answer: {row['correct_idx_qa']}")
    if parsed_response and isinstance(parsed_response, dict):
        print(f"Model Answer: {parsed_response.get('answer')}")
        print(f"Correct: {parsed_response.get('answer') == row['correct_idx_qa']}")
    

def main():
    parser = argparse.ArgumentParser(description='View QA results')
    parser.add_argument('run_id')
    parser.add_argument('record_id')
    args = parser.parse_args()
    
    df = prepare_df(types=['qa'], filter_errors=False, filter_nulls=False)
    
    if args.run_id:
        df = df[df['run_id_qa'] == args.run_id]
    if args.record_id:
        df = df[df['record_id_qa'] == args.record_id]
    
    if len(df) != 1:
        print(f"Error: Expected exactly 1 record, found {len(df)}")
        if not df.empty:
            print("\nMatching records:")
            for _, row in df.iterrows():
                print(f"  Run ID: {row.get('run_id_qa')}, Record ID: {row.get('record_id_qa')}, Question Idx: {row.get('question_idx_qa')}")
        return
    
    row = df.iloc[0]

    display_qa(row)

if __name__ == '__main__':
    main()

