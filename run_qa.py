import os
import re
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from qa_config import (
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT,
    MODEL_NAME,
    NUM_QUESTIONS, RANDOM_SEED, NUM_CHOICES,    
)
from llm_utils import call_openrouter
from dataset_utils import select_questions_and_options, format_options

def load_prompt_template():
    with open('prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts['qa_prompt_template']

def parse_model_response(response_text):
    parsed = {
        'is_valid': False,
        'answer': None,
        'confidence': None,
        'reasoning': None
    }
    
    final_answer_match = re.search(r'<BEGIN FINAL ANSWER>(.*?)</END FINAL ANSWER>', response_text, re.DOTALL | re.IGNORECASE)
    if not final_answer_match:
        return parsed
    
    final_answer_text = final_answer_match.group(1)
    parsed['is_valid'] = True
    
    answer_match = re.search(r'Answer:\s*(\d+)', final_answer_text, re.IGNORECASE)
    if answer_match:
        parsed['answer'] = int(answer_match.group(1))
    
    confidence_match = re.search(r'Confidence:\s*(\d+)(?:\.\d+)?%?', final_answer_text, re.IGNORECASE)
    if confidence_match:
        parsed['confidence'] = int(confidence_match.group(1))
    
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n\s*$|\Z)', final_answer_text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        parsed['reasoning'] = reasoning_match.group(1).strip()
    
    return parsed

def main():
    load_dotenv()    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    print(f"Loading dataset: {DATASET_NAME}/{DATASET_SUBSET}")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    
    print(f"Selecting {NUM_QUESTIONS} questions with seed {RANDOM_SEED}")
    questions_data = select_questions_and_options(DATASET_NAME, dataset, NUM_QUESTIONS, NUM_CHOICES, RANDOM_SEED)
    
    prompt_template = load_prompt_template()

    # Run evaluation
    results = []
    for i, q_data in enumerate(questions_data):
        print(f"Processing question {i+1}/{len(questions_data)}")
        
        options_text = format_options(q_data['options'])
        number_choices = ', '.join(str(i) for i in range(NUM_CHOICES))
        
        prompt = prompt_template.format(
            question=q_data['question'],
            options_text=options_text,
            letter_choices=number_choices,
        )
        
        response = call_openrouter(prompt, MODEL_NAME, api_key)
        
        if 'choices' in response and len(response['choices']) > 0:
            raw_model_response = response['choices'][0]['message']['content']
        else:
            raw_model_response = "Error: No response from model"
        
        parsed_model_response = parse_model_response(raw_model_response)
        
        results.append({
            'question_idx': q_data['original_idx'],
            'question': q_data['question'],
            'options': q_data['options'],
            'correct_idx': q_data['correct_idx'],
            'raw_model_response': raw_model_response,
            'parsed_model_response': parsed_model_response,
            'prompt': prompt
        })
    
    # Save results
    with open('qa_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to qa_results.json")

if __name__ == "__main__":
    main()
