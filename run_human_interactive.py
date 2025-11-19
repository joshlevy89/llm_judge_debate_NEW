from utils.debate_utils import *
import argparse
import time

def __main__():
    import os
import re
import json
from pandas.core.strings.accessor import NoNewAttributesMixin
import yaml
import random
import string
import time
import traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
import config.config_debate as config_debate
from config.config_debate import *
from utils.dataset_utils import select_questions_and_options, format_options
from utils.debate_utils import *
from utils.shared_utils import extract_config, generate_run_id, load_prompts, format_latex
import os

RESPONSE_DURATION_MASKING = None

def setup_output_path():
    output_dir = Path('results') / 'human'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'human_interactive_debate.jsonl'

def get_human_action():
    # Get the next action from the human
    while True:
        print("="*80)
        print("Actions: 'next', 'end', or '<debater_idx>: <message>'")
        print("="*80)
        message = input("> ").strip()
        
        if message == 'end':
            action, new_debater_idx = message, None
            break
        elif message == 'next':
            action, new_debater_idx = message, None
            break
        elif ':' in message:
            parts = message.split(':', 1)
            try:
                action, new_debater_idx = message, int(parts[0].strip())
                break
            except ValueError:
                print("Invalid format. Use: <debater_idx>: <message>")
        else:
            print("Invalid action.")
    return action, new_debater_idx


def main():
    # Add  
    parser = argparse.ArgumentParser()
    parser.add_argument('question_idx', type=int)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    run_id = generate_run_id()
    run_datetime = datetime.now().isoformat()
    results_path = setup_output_path()
    config = extract_config(config_debate)
    
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    questions_data = select_questions_and_options(DATASET_NAME, dataset, 1, NUM_CHOICES, None, [args.question_idx])
    
    debater_template, private_reasoning_prompt, _ = load_prompts()
    
    for i, q_data in enumerate(questions_data):
        
        record_id = generate_run_id()
        debate_history = []

        # DISPLAY THE QUESTION
        print('='*80)
        print(format_latex(q_data['question']))
        for idx, opt in enumerate(q_data['options']):
            print(f"{idx}: {format_latex(opt)}")

        # RUN DEBATE
        start_debate_time = time.time()
        debater_assignments = q_data['options']
        cur_debater_idx = -1

        for turn in range(NUM_TURNS):

            # get human action 
            action, new_debater_idx = get_human_action()
            debate_history.append({
                'persona': 'judge',
                'action': action,
                'is_human': True
            })
            if action == 'end':
                break
            if new_debater_idx is not None:
                cur_debater_idx = new_debater_idx
            else:
                cur_debater_idx += 1
                cur_debater_idx = cur_debater_idx % len(debater_assignments) # cycle back

            # run a turn of debate
            while True:
                start_turn_time = time.time()
                turn_response = run_debate_turn(turn, debater_assignments, cur_debater_idx, q_data['question'], debate_history, debater_template, private_reasoning_prompt, api_key, run_id, record_id, mock=MOCK_DEBATE_RESPONSE)
                
                print(f"{'='*80}\nDebater {turn_response['debater_idx']} (Turn {turn_response['turn']})\n{'='*80}\n")
                if turn_response['success']:
                    turn_duration = time.time() - start_turn_time
                    if RESPONSE_DURATION_MASKING is not None and turn_duration < RESPONSE_DURATION_MASKING:
                        # This is to mask which debater is responding (as the incorrect debater usuallly takes longer)
                        time.sleep(RESPONSE_DURATION_MASKING - turn_duration)
                    print(format_latex(turn_response['parsed_response']['public_argument']))
                    debate_history.append(turn_response)                
                    break
                else:
                    # re-run the turn
                    print('ERROR ' * 80)
                    print('RE-RUNNING TURN AND NOT APPENDING TO DEBATE_HISTORY')
                    continue

        debate_duration = time.time() - start_debate_time

        print("="*80)

        print("Verdict (index): ")
        verdict = input("> ").strip()
        
        print("Confidence (0-100): ")
        confidence = input("> ").strip()

        print("Reasoning: ")
        reasoning = input("> ").strip()

        print(f"Correct Idx {q_data['correct_idx']}")

        question_result = {
            'run_id': run_id,
            'record_id': record_id,
            'datetime': run_datetime,
            'config': config,
            'prompt_template': {'debater_prompt_template': debater_template, 'private_reasoning_template': private_reasoning_prompt if PRIVATE_SCRATCHPAD else None},
            'question_idx': q_data['original_idx'],
            'question': q_data['question'],
            'options': q_data['options'],
            'correct_idx': q_data['correct_idx']
            }

        question_result['debate_duration'] = debate_duration
        question_result['success'] = 'success'
        question_result['error_message'] = None
        question_result['debate_history'] = debate_history
        question_result['verdict'] = verdict
        question_result['confidence'] = confidence
        question_result['reasoning'] = reasoning

        # Save the data 
        with open(results_path, 'a') as f:
            f.write(json.dumps(question_result) + '\n')
            f.flush()


if __name__ == "__main__":
    main()

