import pandas as pd

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
            'verdict_minus_judge_qa': group_df['is_correct_verdict'].mean() - group_df['is_correct_qa_judge'].mean(),
            'pgr': (group_df['is_correct_verdict'].mean() - group_df['is_correct_qa_judge'].mean()) / (group_df['is_correct_qa_debater'].mean() - group_df['is_correct_qa_judge'].mean()),
            'debater_minus_judge_qa': group_df['is_correct_qa_debater'].mean() - group_df['is_correct_qa_judge'].mean(),
        })
    
    return pd.DataFrame(results)