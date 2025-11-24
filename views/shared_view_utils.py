from views.view_qa import display_qa
from analysis.analysis_utils import prepare_df

def find_and_display_qa(row):
    row['options_str'] = str(row['options'])

    if 'judge_model' in row['config']:
        qa_filters = {
            'question': row['question'],
            'options': row['options'],
            'config': {'model_name': row['config']['judge_model']}
        }
        qa_df = prepare_df(types=['qa'], filter_errors=False, filter_nulls=False, qa_filters=qa_filters)
        qa_judge_row = qa_df.iloc[0]
        # qa_debater_row = qa_df[(qa_df['question_qa_judge'] == row['question_qa_debater']) & (qa_df['options_str_qa_judge'] == row['options_str_qa_debater']) & (qa_df['config_model_name_qa_judge'] == row['config_debater_model_debates'])].iloc[0]
        
        print('='*80)
        print('Direct Judge QA')
        print('='*80)
        display_qa(qa_judge_row, display_question=False)
    
    if 'debater_model' in row['config']:
        qa_filters = {
            'question': row['question'],
            'options': row['options'],
            'config': {'model_name': row['config']['debater_model']}
        }
        qa_df = prepare_df(types=['qa'], filter_errors=False, filter_nulls=False, qa_filters=qa_filters)
        qa_judge_row = qa_df.iloc[0]
        # qa_debater_row = qa_df[(qa_df['question_qa_judge'] == row['question_qa_debater']) & (qa_df['options_str_qa_judge'] == row['options_str_qa_debater']) & (qa_df['config_model_name_qa_judge'] == row['config_debater_model_debates'])].iloc[0]
        
        print('='*80)
        print(f"Direct Debater QA ({row['config']['debater_model']})")
        print('='*80)
        display_qa(qa_judge_row, display_question=False)
    # display_qa(qa_debater_row)