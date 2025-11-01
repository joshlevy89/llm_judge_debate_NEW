import pandas as pd

def load_qa(filters=None):
    df = pd.read_json("results/qa/qa_results.jsonl", lines=True)
    
    df = df[df['parsed_model_response'].apply(
        lambda x: isinstance(x, dict) and x.get('is_valid', False) and 'answer' in x
    )].reset_index(drop=True)
    
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
    
    print(f"starting records: {start_count}")
    if judge_missing > 0:
        print(f"dropped {judge_missing} without judge QA")
    if debater_missing > 0:
        print(f"dropped {debater_missing} without debater QA")
    print(f"final records: {len(df)}")
    
    return df

