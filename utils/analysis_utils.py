import pandas as pd
from typing import Literal

def load_qa(filters=None):
    df = pd.read_json("results/qa/qa_results.jsonl", lines=True)
    
    df = df[
        df['parsed_model_response'].apply(
            lambda x: isinstance(x, dict) and x.get('is_valid', False) and 'answer' in x
        ) & (df['success'].fillna(True))
    ].reset_index(drop=True)
    
    config_df = pd.json_normalize(df['config'])
    config_df.columns = ['config_' + col for col in config_df.columns]
    df = pd.concat([df, config_df], axis=1)

    df['options_str'] = df['options'].apply(str)

    df = df.sort_values('datetime', ascending=False).drop_duplicates(
        subset=['question', 'options_str', 'config_model_name'], keep='first')

    df['parsed_answer'] = df['parsed_model_response'].apply(lambda x: x['answer'])
    df = df[df['parsed_answer'].notna()]

    df['is_correct'] = df['parsed_answer'] == df['correct_idx']
    
    if filters:
        for col, val in filters.items():
            if val is None:
                df = df[df[col].isna()]
            else:
                df = df[df[col] == val]
    
    return df


def load_debate(debate_run_id):
    df = pd.read_json(f"results/debates/{debate_run_id}.jsonl", lines=True)
    config_df = pd.json_normalize(df['config'])
    config_df.columns = ['config_debate_' + col for col in config_df.columns]
    df = pd.concat([df, config_df], axis=1)
    df['options_str'] = df['options'].apply(str)
    
    if 'config_debate_private_reasoning_word_limit' in df.columns:
        df['config_debate_private_reasoning_word_limit'] = df['config_debate_private_reasoning_word_limit'].astype('object')
        df.loc[df['config_debate_private_scratchpad'] == False, 'config_debate_private_reasoning_word_limit'] = None
    
    return df


def load_verdict(verdict_run_id):
    df = pd.read_json(f"results/verdicts/{verdict_run_id}.jsonl", lines=True)

    df = df[df['judge_verdict'].apply(
        lambda x: isinstance(x, dict) and isinstance(x.get('parsed'), dict) and 'answer' in x.get('parsed', {})
    )].reset_index(drop=True)
    
    config_df = pd.json_normalize(df['config'])
    config_df.columns = ['config_verdict_' + col for col in config_df.columns]
    df = pd.concat([df, config_df], axis=1)
    
    df['parsed_answer'] = df['judge_verdict'].apply(lambda x: x['parsed']['answer'])
    df = df[df['parsed_answer'].notna()]
    df['is_correct_verdict'] = df['parsed_answer'] == df['correct_idx']
    
    return df


def load_debate_and_verdict(verdict_run_id):
    verdict_df = load_verdict(verdict_run_id)
    debate_run_id = verdict_df['config_verdict_debate_run_id'].iloc[0]
    debate_df = load_debate(debate_run_id)
    
    verdict_cols = ['record_id', 'verdict_run_id', 'parsed_answer', 'is_correct_verdict'] + \
                   [col for col in verdict_df.columns if col.startswith('config_verdict_')]
    df = debate_df.merge(verdict_df[verdict_cols], on='record_id')
    
    return df


def load_debate_and_verdict_and_qa(verdict_run_id):
    df = load_debate_and_verdict(verdict_run_id)
    start_count = len(df)
    
    qa_df = load_qa(filters=None)
    qa_df = qa_df[['question', 'options_str', 'config_model_name', 'is_correct']]
    
    judge_qa = qa_df.rename(columns={'is_correct': 'is_correct_judge_qa'})
    df = df.merge(judge_qa, left_on=['question', 'options_str', 'config_verdict_judge_model'], 
                  right_on=['question', 'options_str', 'config_model_name'], how='left')
    judge_missing = df['is_correct_judge_qa'].isna().sum()
    
    debater_qa = qa_df.rename(columns={'is_correct': 'is_correct_debater_qa'})
    df = df.merge(debater_qa, left_on=['question', 'options_str', 'config_debate_debater_model'], 
                  right_on=['question', 'options_str', 'config_model_name'], how='left')
    debater_missing = df['is_correct_debater_qa'].isna().sum()
    
    df = df[df['is_correct_verdict'].notna() & df['is_correct_judge_qa'].notna() & df['is_correct_debater_qa'].notna()]
    
    print(f"loaded verdict: {verdict_run_id}: starting records: {start_count}, dropped {judge_missing} without judge QA, dropped {debater_missing} without debater QA, final records: {len(df)}")
    
    return df



def load_unique_over_runs(run_ids, type: Literal['load_debate_and_verdict_and_qa']):
    if type == 'load_debate_and_verdict_and_qa':
        # marginalizes over the verdict run id
        fn = load_debate_and_verdict_and_qa
        dedupe_columns = [
            "record_id",
            "config_verdict_debate_run_id",
            "config_debate_dataset_name", 
            "config_debate_dataset_subset", 
            "config_debate_dataset_split", 
            "config_debate_debater_model", 
            "config_debate_debater_temperature", 
            "config_debate_random_seed", 
            "config_debate_num_choices", 
            "config_debate_num_turns", 
            "config_debate_private_scratchpad",
            "config_debate_public_argument_word_limit",
            "config_debate_private_reasoning_word_limit",
            "config_verdict_judge_model", 
            "config_verdict_judge_temperature", 
            "config_verdict_max_output_tokens"
        ]
    
    master_df = pd.DataFrame()
    for run_id in run_ids:
        df = fn(run_id)
        master_df = pd.concat([master_df, df])
    
    master_df = master_df.sort_values('datetime').drop_duplicates(subset=dedupe_columns, keep='last')

    return master_df, dedupe_columns


