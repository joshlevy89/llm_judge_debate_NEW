import pandas as pd
import glob
from pathlib import Path

def get_project_root():
    return Path(__file__).parent.parent

def extract_parsed_answer(df, type):
    if type == 'verdicts':
        return df['judge_verdict'].apply(lambda x: x.get('parsed', {}).get('answer') if pd.notna(x) and isinstance(x, dict) else None)
    elif type == 'qa':
        return df['parsed_model_response'].apply(lambda x: x.get('answer') if pd.notna(x) else None)
    return None


def load_all_records_into_df(type, filter_errors=True, filter_nulls=True):
    results_dir = get_project_root() / 'results'
    files = glob.glob(str(results_dir / type / '*.jsonl'))
    dfs = []
    for file in files:
        df = pd.read_json(file, lines=True)
        if df.shape[0] == 0:
            continue

        config_df = pd.json_normalize(df['config'])
        config_df.columns = ['config_' + col for col in config_df.columns]
        df = pd.concat([df.drop(columns=['config']), config_df], axis=1)

        df['options_str'] = df['options'].apply(str)

        parsed_answer = extract_parsed_answer(df, type)
        if parsed_answer is not None:
            df['parsed_answer'] = parsed_answer
        
        if filter_errors and 'success' in df.columns:
            # Keep records that succeeded OR where success is NaN (legacy records before this field existed)
            df = df[(df['success'] == True) | (df['success'].isna())]
        
        if filter_nulls and 'parsed_answer' in df.columns:
            df = df[df['parsed_answer'].notna()]
        
        df.columns = [col + '_' + type for col in df.columns]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    
    if len(dfs) == 1:
        return dfs[0]
    
    all_columns = sorted(set().union(*[df.columns for df in dfs]))
    aligned_dfs = [df.reindex(columns=all_columns) for df in dfs]
    return pd.concat(aligned_dfs, ignore_index=True)


def prepare_df(types=['verdicts', 'debates', 'qa'], filter_errors=True, filter_nulls=True):
    if isinstance(types, str):
        types = [types]
    
    if types == ['qa']:
        return load_all_records_into_df('qa', filter_errors=filter_errors, filter_nulls=filter_nulls)
    
    if types == ['debates']:
        return load_all_records_into_df('debates', filter_errors=filter_errors, filter_nulls=filter_nulls)

    verdict_df = load_all_records_into_df('verdicts', filter_errors=filter_errors, filter_nulls=filter_nulls)
    debate_df = load_all_records_into_df('debates', filter_errors=filter_errors, filter_nulls=filter_nulls)
    
    if types == ['verdicts']:
        valid_verdicts = verdict_df[verdict_df['record_id_verdicts'].notna()]
        valid_debates = debate_df[debate_df['record_id_debates'].notna()]
        merged = valid_verdicts.merge(valid_debates, left_on=['record_id_verdicts'], right_on=['record_id_debates'], how='left')
        
        invalid_verdicts = verdict_df[verdict_df['record_id_verdicts'].isna()]
        if not invalid_verdicts.empty:
            merged = pd.concat([merged, invalid_verdicts], ignore_index=True, sort=False)
        
        return merged
    
    qa_df = load_all_records_into_df('qa', filter_errors=filter_errors, filter_nulls=filter_nulls)
    
    # Keep only most recent QA record for each (question, options, model) triplet
    if 'datetime_qa' in qa_df.columns:
        qa_df = qa_df.sort_values('datetime_qa', ascending=False)
    qa_df = qa_df.drop_duplicates(subset=['question_qa', 'options_str_qa', 'config_model_name_qa'], keep='first')
    
    valid_verdicts = verdict_df[verdict_df['record_id_verdicts'].notna()]
    valid_debates = debate_df[debate_df['record_id_debates'].notna()]
    verdict_and_debate_df = valid_verdicts.merge(valid_debates, left_on=['record_id_verdicts'], right_on=['record_id_debates'], how='left')

    judge_qa_df = qa_df.copy()
    judge_qa_df.columns = [col + '_judge' for col in qa_df.columns]

    debater_qa_df = qa_df.copy()
    debater_qa_df.columns = [col + '_debater' for col in qa_df.columns]


    all_df = verdict_and_debate_df.merge(
        judge_qa_df[['question_qa_judge', 'options_str_qa_judge', 'config_model_name_qa_judge', 'parsed_answer_qa_judge', 'success_qa_judge']], 
        left_on=['question_verdicts', 'options_str_verdicts', 'config_judge_model_verdicts'], 
        right_on=['question_qa_judge', 'options_str_qa_judge', 'config_model_name_qa_judge'],
        how='left'
    )

    all_df = all_df.merge(
        debater_qa_df[['question_qa_debater', 'options_str_qa_debater', 'config_model_name_qa_debater', 'parsed_answer_qa_debater', 'success_qa_debater']], 
        left_on=['question_verdicts', 'options_str_verdicts', 'config_debater_model_debates'], 
        right_on=['question_qa_debater', 'options_str_qa_debater', 'config_model_name_qa_debater'],
        how='left',
        suffixes=('', '_debater')
    )

    # Filter to only valid records if requested (default for analysis)
    # Note: success can be NaN after left joins if matching records weren't found,
    # or for legacy data that predates the success field
    if filter_errors and filter_nulls:
        all_df = all_df[
            ((all_df['success_verdicts'] == True) | (all_df['success_verdicts'].isna())) & 
            ((all_df['success_debates'] == True) | (all_df['success_debates'].isna())) & 
            ((all_df['success_qa_judge'] == True) | (all_df['success_qa_judge'].isna())) & 
            ((all_df['success_qa_debater'] == True) | (all_df['success_qa_debater'].isna())) &
            all_df['parsed_answer_qa_judge'].notnull() & 
            all_df['parsed_answer_qa_debater'].notnull() & 
            all_df['parsed_answer_verdicts'].notnull()
        ]

    # Add correctness columns
    all_df['is_correct_qa_judge'] = all_df['parsed_answer_qa_judge'] == all_df['correct_idx_verdicts']
    all_df['is_correct_qa_debater'] = all_df['parsed_answer_qa_debater'] == all_df['correct_idx_verdicts']
    all_df['is_correct_verdict'] = all_df['parsed_answer_verdicts'] == all_df['correct_idx_verdicts']

    return all_df


def aggregate_by_fields(input_df, fields):
    results = []
    
    for group_vals, group_df in input_df.groupby(fields):
        name = ", ".join([str(v) for v in (group_vals if isinstance(group_vals, tuple) else (group_vals,))])
        
        results.append({
            'name': name,
            'debater_qa_acc': group_df['is_correct_qa_debater'].mean(),
            'judge_qa_acc': group_df['is_correct_qa_judge'].mean(),
            'verdict_acc': group_df['is_correct_verdict'].mean(),
            'debater_qa_n_correct': group_df['is_correct_qa_debater'].sum(),
            'judge_qa_n_correct': group_df['is_correct_qa_judge'].sum(),
            'verdict_n_correct': group_df['is_correct_verdict'].sum(),
            'n_total': len(group_df),
            'pgr': (group_df['is_correct_verdict'].mean() - group_df['is_correct_qa_judge'].mean()) / (group_df['is_correct_qa_debater'].mean() - group_df['is_correct_qa_judge'].mean()),
            'gap': group_df['is_correct_qa_debater'].mean() - group_df['is_correct_qa_judge'].mean(),
            'gain':group_df['is_correct_verdict'].mean() - group_df['is_correct_qa_judge'].mean()
        })
    
    return pd.DataFrame(results)