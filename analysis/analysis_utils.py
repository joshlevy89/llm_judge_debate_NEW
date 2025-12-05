import pandas as pd
import glob
import json
from pathlib import Path

def get_project_root():
    return Path(__file__).parent.parent

def matches_filters(data, filters):
    for key, filter_val in filters.items():
        data_val = data.get(key, {}) if isinstance(filter_val, dict) else data.get(key)
        matches = all(data_val.get(k) == v for k, v in filter_val.items()) if isinstance(filter_val, dict) else data_val == filter_val
        if not matches:
            return False
    return True

def extract_parsed_answer(df, type):
    if type == 'verdicts':
        if 'judge_verdict' not in df.columns:
            return None
        return df['judge_verdict'].apply(lambda x: x.get('parsed', {}).get('answer') if pd.notna(x) and isinstance(x, dict) else None)
    elif type == 'qa':
        return df['parsed_model_response'].apply(lambda x: x.get('answer') if pd.notna(x) else None)
    return None


def load_all_records_into_df(type, filter_errors=True, filter_nulls=True, qa_filters=None, specific_verdict_ids=None):
    results_dir = get_project_root() / 'results'
    if type == 'human':
        files = [str(results_dir / type / 'human_interactive_debate.jsonl')]
    else:
        files = glob.glob(str(results_dir / type / '*.jsonl'))
    dfs = []

    if type == 'verdicts' and specific_verdict_ids is not None:
        # filter to just those verdict ids
        files = [f for f in files if any(vid in f for vid in specific_verdict_ids)]

    for file in files:
        if type == 'qa' and qa_filters is not None:
            records = []
            for line in open(file):
                data = json.loads(line)
                if matches_filters(data, qa_filters):
                    records.append(data)
            if not records:
                continue
            
            df = pd.DataFrame(records)
        else:
            df = pd.read_json(file, lines=True)

        if df.shape[0] == 0:
            continue

        # for debugging purposes, add a few lines of code where only keep qa results from runs before 11-05-2025 (nov 5 2025)
        # if type == 'qa' and 'datetime' in df.columns:
        #     print('before')
        #     print(df.shape)
        #     cutoff_date = pd.Timestamp('2025-11-07')
        #     df['datetime'] = pd.to_datetime(df['datetime'])
        #     df = df[df['datetime'] < cutoff_date]
        #     print('after')
        #     print(df.shape)

        config_df = pd.json_normalize(df['config'])
        config_df.columns = ['config_' + col for col in config_df.columns]
        df = pd.concat([df.drop(columns=['config']), config_df], axis=1)

        df['options_str'] = df['options'].apply(str)

        if type == 'verdicts':
            if 'judge_verdict' not in df.columns:
                df['parsed_answer'] = None
                df['parsed_confidence'] = None
                df['parsed_reasoning'] = None
            else:
                df['parsed_answer'] = df['judge_verdict'].apply(lambda x: x.get('parsed', {}).get('answer') if pd.notna(x) and isinstance(x, dict) else None)
                df['parsed_confidence'] = df['judge_verdict'].apply(lambda x: x.get('parsed', {}).get('confidence') if pd.notna(x) and isinstance(x, dict) else None)
                df['parsed_reasoning'] = df['judge_verdict'].apply(lambda x: x.get('parsed', {}).get('reasoning') if pd.notna(x) and isinstance(x, dict) else None)
        elif type == 'qa':
            df['parsed_answer'] = df['parsed_model_response'].apply(lambda x: x.get('answer') if pd.notna(x) else None)
        
        if filter_errors and 'success' in df.columns:
            # Keep records that succeeded OR where success is NaN (legacy records before this field existed)
            df = df[(df['success'] == True) | (df['success'].isna()) | (df['success'] == 'success')]
        
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


def prepare_df(types=['verdicts', 'debates', 'qa'], filter_errors=True, filter_nulls=True, qa_filters=None, specific_verdict_ids=None):
    if isinstance(types, str):
        types = [types]
    
    if 'qa' in types:
        qa_df = load_all_records_into_df('qa', filter_errors=filter_errors, filter_nulls=filter_nulls, qa_filters=qa_filters)
        # Keep only most recent QA record for each (question, options, model) triplet
        if 'datetime_qa' in qa_df.columns:
            qa_df = qa_df.sort_values('datetime_qa', ascending=False)
        qa_df = qa_df.drop_duplicates(subset=['question_qa', 'options_str_qa', 'config_model_name_qa'], keep='first')
        if qa_df.empty and qa_filters is not None:
            raise Exception(f'There are no QA records with the provided filter: {qa_filters}')
        qa_df['is_correct_qa'] = qa_df['parsed_answer_qa'] == qa_df['correct_idx_qa']
        if filter_errors:
            qa_df = qa_df[((qa_df['success_qa'] == True) | (qa_df['success_qa'].isna()))]
        if filter_nulls:
            qa_df = qa_df[qa_df['parsed_answer_qa'].notnull()]

        if types == ['qa']:
            return qa_df


    if 'debates' in types or 'verdicts' in types or 'human' in types:
        if 'human' in types:
            debate_df = load_all_records_into_df('human', filter_errors=filter_errors, filter_nulls=filter_nulls)
            debate_df.columns = [name.replace('_human', '_debates') if '_human' in name else name for name in debate_df.columns]
        else:
            debate_df = load_all_records_into_df('debates', filter_errors=filter_errors, filter_nulls=filter_nulls)
            debate_df = debate_df.drop(columns=['config_judge_model_debates']) # this field shouldn't exist
        if types == ['debates'] or types == ['human']:
            return debate_df

    verdict_df = load_all_records_into_df('verdicts', filter_errors=filter_errors, filter_nulls=filter_nulls, specific_verdict_ids=specific_verdict_ids)
    
    if types == ['verdicts']:
        valid_verdicts = verdict_df[verdict_df['record_id_verdicts'].notna()]
        valid_debates = debate_df[debate_df['record_id_debates'].notna()]
        merged = valid_verdicts.merge(valid_debates, left_on=['record_id_verdicts'], right_on=['record_id_debates'], how='left')
        
        invalid_verdicts = verdict_df[verdict_df['record_id_verdicts'].isna()]
        if not invalid_verdicts.empty:
            merged = pd.concat([merged, invalid_verdicts], ignore_index=True, sort=False)
        
        return merged

    valid_verdicts = verdict_df[verdict_df['record_id_verdicts'].notna()]
    valid_debates = debate_df[debate_df['record_id_debates'].notna()]
    verdict_and_debate_df = valid_verdicts.merge(valid_debates, left_on=['record_id_verdicts'], right_on=['record_id_debates'], how='left')

    if 'qa' in types:
        judge_qa_df = qa_df.copy()
        judge_qa_df.columns = [col + '_judge' for col in qa_df.columns]

        debater_qa_df = qa_df.copy()
        debater_qa_df.columns = [col + '_debater' for col in qa_df.columns]

        all_df = verdict_and_debate_df.merge(
            judge_qa_df[['question_qa_judge', 'options_str_qa_judge', 'config_model_name_qa_judge', 'parsed_answer_qa_judge', 'is_correct_qa_judge', 'success_qa_judge']], 
            left_on=['question_verdicts', 'options_str_verdicts', 'config_judge_model_verdicts'], 
            right_on=['question_qa_judge', 'options_str_qa_judge', 'config_model_name_qa_judge'],
            how='left')

        all_df = all_df.merge(
            debater_qa_df[['question_qa_debater', 'options_str_qa_debater', 'config_model_name_qa_debater', 'parsed_answer_qa_debater', 'is_correct_qa_debater', 'success_qa_debater']], 
            left_on=['question_verdicts', 'options_str_verdicts', 'config_debater_model_debates'], 
            right_on=['question_qa_debater', 'options_str_qa_debater', 'config_model_name_qa_debater'],
            how='left',
            suffixes=('', '_debater'))
    else:
        all_df = verdict_and_debate_df

    # Filter to only valid records if requested (default for analysis)
    # Note: success can be NaN after left joins if matching records weren't found,
    # or for legacy data that predates the success field
    if filter_errors:
        all_df = all_df[
            ((all_df['success_verdicts'] == True) | (all_df['success_verdicts'].isna()) | (all_df['success_verdicts'] == 'success')) & 
            ((all_df['success_debates'] == True) | (all_df['success_debates'].isna()) | (all_df['success_debates'] == 'success') ) 
        ]
    if filter_nulls:
        all_df = all_df[all_df['parsed_answer_verdicts'].notnull()]

    all_df['is_correct_verdict'] = all_df['parsed_answer_verdicts'] == all_df['correct_idx_verdicts']
    # Get rid of any answers that are longer than the options list
    all_df['parsed_answer_verdicts'] = all_df['parsed_answer_verdicts'].astype(int)
    all_df = all_df[all_df['parsed_answer_verdicts'] < all_df['config_num_choices_debates']]

    return all_df

def aggregate_acc(df):

    result = ({
        'debater_qa_acc': df['is_correct_qa_debater'].mean(),
        'judge_qa_acc': df['is_correct_qa_judge'].mean(),
        'verdict_acc': df['is_correct_verdict'].mean(),
        'debater_qa_n_correct': df['is_correct_qa_debater'].sum(),
        'judge_qa_n_correct': df['is_correct_qa_judge'].sum(),
        'verdict_n_correct': df['is_correct_verdict'].sum(),
        'n_total': len(df),
        'n_verdict_not_null': df['is_correct_verdict'].notnull().sum(), 
        'n_judge_qa_not_null': df['is_correct_qa_judge'].notnull().sum(), 
        'n_debater_qa_not_null': df['is_correct_qa_debater'].notnull().sum(), 
        'verdict_chose_idx_0': (df['parsed_answer_verdicts'] == 0).mean(),
        'pgr': (df['is_correct_verdict'].mean() - df['is_correct_qa_judge'].mean()) / (df['is_correct_qa_debater'].mean() - df['is_correct_qa_judge'].mean()),
        'gap': df['is_correct_qa_debater'].mean() - df['is_correct_qa_judge'].mean(),
        'gain':df['is_correct_verdict'].mean() - df['is_correct_qa_judge'].mean()
    })

    return result

def aggregate_by_fields(input_df, fields):
    results = []
    
    for group_vals, group_df in input_df.groupby(fields):
        name = ", ".join([str(v) for v in (group_vals if isinstance(group_vals, tuple) else (group_vals,))])
        
        result = aggregate_acc(group_df)
        result['name'] = name
        results.append(result)
    
    return pd.DataFrame(results)


def sum_reasoning_tokens_over_turns(turns):
    tot = 0
    for turn in turns:
        # if turn['token_usage'].get('completion_tokens_details') is None:
        #     continue
        reasoning_tokens = get_reasoning_tokens(turn)
        tot += reasoning_tokens
    return tot

def get_reasoning_tokens(x):
    completion_tokens_details = x['token_usage'].get('completion_tokens_details')
    if completion_tokens_details is None:
        return 0
    else:
        return completion_tokens_details.get('reasoning_tokens', 0)