import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
from typing import Literal
from typing import Literal
import scipy
from scipy import stats
from scipy import stats
from sklearn.linear_model import LinearRegression

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

def load_all_over_runs(run_ids, type: Literal['load_debate_and_verdict_and_qa']):
    if type == 'load_debate_and_verdict_and_qa':
        # marginalizes over the verdict run id
        fn = load_debate_and_verdict_and_qa
    
    master_df = pd.DataFrame()
    for run_id in run_ids:
        df = fn(run_id)
        master_df = pd.concat([master_df, df])
    return master_df


def load_unique_over_runs(run_ids, type: Literal['load_debate_and_verdict_and_qa']):
    if type == 'load_debate_and_verdict_and_qa':
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
    
    master_df = load_all_over_runs(run_ids, type)
    master_df = master_df.sort_values('datetime').drop_duplicates(subset=dedupe_columns, keep='last')

    dedupe_columns = [col for col in dedupe_columns if col != 'record_id']
    unique_configs = master_df[dedupe_columns].drop_duplicates().to_dict('records')
    unique_configs_df = pd.DataFrame(unique_configs).reset_index().rename(columns={'index': 'unique_config_id'})
    master_df = master_df.merge(unique_configs_df, on=dedupe_columns, how='left')

    return master_df, unique_configs


def sort_and_color_by_model_family(names):
    def normalize_model_name(text):
        return re.sub(r'([a-zA-Z])(\d)', r'\1-\2', text)
    
    def natural_sort_key(text):
        normalized = normalize_model_name(text)
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', normalized)]
    
    models_df = pd.DataFrame({'name': names})
    models_df['family'] = models_df['name'].str.split('/').str[0]
    models_df['model'] = models_df['name'].str.split('/').str[1]
    models_df['sort_key'] = models_df.apply(lambda row: (row['family'], natural_sort_key(row['model'])), axis=1)
    models_df = models_df.sort_values('sort_key').reset_index(drop=True)
    
    families = models_df['family'].unique()
    n_families = len(families)
    base_colors = plt.cm.tab10(np.linspace(0, 1, n_families)) if n_families <= 10 else plt.cm.hsv(np.linspace(0, 0.9, n_families))
    
    color_map = {}
    for family_idx, family in enumerate(families):
        family_rows = models_df[models_df['family'] == family]
        n_models = len(family_rows)
        base_color = base_colors[family_idx][:3]
        
        for model_idx, name in enumerate(family_rows['name']):
            lightness = 0.4 + (0.6 * model_idx / max(1, n_models - 1))
            color = mcolors.to_rgb(base_color)
            adjusted_color = tuple(c * lightness + (1 - lightness) for c in color)
            color_map[name] = adjusted_color
    
    return models_df['name'].tolist(), color_map


def _prep_results_df(results_df):
    results_df = results_df.copy()
    results_df['verdict_minus_judge_qa'] = results_df['verdict_acc'] - results_df['judge_qa_acc']
    names, _ = sort_and_color_by_model_family(results_df['name'].unique())
    return results_df.set_index('name').loc[names].reset_index()

def plot_accuracy_bars(results_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 6))
    
    results_df = _prep_results_df(results_df)
    x = np.arange(len(results_df))
    width = 0.25

    bars1 = ax.bar(x - width, results_df['debater_qa_acc'], width, label='Debater QA', alpha=0.8)
    bars2 = ax.bar(x, results_df['judge_qa_acc'], width, label='Judge QA', alpha=0.8)
    bars3 = ax.bar(x + width, results_df['verdict_acc'], width, label='Verdict', alpha=0.8)

    for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
        ratio1 = round(results_df.iloc[i]['debater_qa_acc'], 2)
        ratio2 = round(results_df.iloc[i]['judge_qa_acc'], 2)
        ratio3 = round(results_df.iloc[i]['verdict_acc'], 2)
        
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02, 
                f"{ratio1}\n{results_df.iloc[i]['debater_qa_n_correct']:.0f}/{results_df.iloc[i]['n_total']}", 
                ha='center', va='bottom', fontsize=9)
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02, 
                f"{ratio2}\n{results_df.iloc[i]['judge_qa_n_correct']:.0f}/{results_df.iloc[i]['n_total']}", 
                ha='center', va='bottom', fontsize=9)
        ax.text(b3.get_x() + b3.get_width()/2, b3.get_height() + 0.02, 
                f"{ratio3}\n{results_df.iloc[i]['verdict_n_correct']:.0f}/{results_df.iloc[i]['n_total']}", 
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['name'], rotation=45, ha='right')
    return ax

def plot_verdict_difference(results_df, ax=None, type='gain'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 2))
    
    results_df = _prep_results_df(results_df)
    if type == 'gain':
        bars =ax.bar(results_df['name'], results_df['verdict_minus_judge_qa'], color='purple')
        ylabel = 'Gain'
    elif type == 'pgr':
        bars = ax.bar(results_df['name'], results_df['pgr'], color='purple')
        ylabel = 'PGR'
    elif type == 'gap':
        bars = ax.bar(results_df['name'], results_df['debater_minus_judge_qa'], color='purple')
        ylabel = 'Gap'
    ax.set_ylim(-.1, 0.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Judge Model')
    ax.set_xticklabels(results_df['name'], rotation=45, ha='right')
    return bars

def plot_results_by_name(results_df):
    fig, ax = plt.subplots(2, 1, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plot_accuracy_bars(results_df, ax=ax[0])
    plot_verdict_difference(results_df, ax=ax[1])
    plt.tight_layout()
    plt.show()


def results_by_config(unique_df, unique_configs):
    """
    Collapses across runs to get the unique records pertaining to each config (or, the config params that matter)
    """
    results = []
    varying_cols = [col for col in unique_configs[0].keys() if unique_df[col].nunique() > 1]

    # Get the records and summary for each unique config
    for unique_config in unique_configs:
        filtered_df = unique_df.copy()
        for key, value in unique_config.items():
            if value is None or pd.isna(value):
                filtered_df = filtered_df[filtered_df[key].isna()]
            else:
                filtered_df = filtered_df[filtered_df[key] == value]

        # Name the column for the value of the varying columns
        name = ""
        for col in varying_cols:
            if col in unique_config:
                name += (f"{unique_config[col]},")
        name = name.strip(",")

        results.append({
            'name': name,
            'debater_qa_acc': filtered_df['is_correct_debater_qa'].mean(),
            'judge_qa_acc': filtered_df['is_correct_judge_qa'].mean(),
            'verdict_acc': filtered_df['is_correct_verdict'].mean(),
            'debater_qa_n_correct': filtered_df['is_correct_debater_qa'].sum(),
            'judge_qa_n_correct': filtered_df['is_correct_judge_qa'].sum(),
            'verdict_n_correct': filtered_df['is_correct_verdict'].sum(),
            'n_total': len(filtered_df),
            'verdict_minus_judge_qa': filtered_df['is_correct_verdict'].mean() - filtered_df['is_correct_judge_qa'].mean(),
            'pgr': (filtered_df['is_correct_verdict'].mean() - filtered_df['is_correct_judge_qa'].mean()) / (filtered_df['is_correct_debater_qa'].mean() - filtered_df['is_correct_judge_qa'].mean()),
            'debater_minus_judge_qa': filtered_df['is_correct_debater_qa'].mean() - filtered_df['is_correct_judge_qa'].mean(),
        })

    results_df = pd.DataFrame(results)
    return results_df

def results_by_run(verdict_ids):
    results = []
    for vid in verdict_ids:
        df = load_debate_and_verdict_and_qa(vid)
        n = len(df)
        results.append({
            'name': vid,
            'debater_qa_acc': df['is_correct_debater_qa'].mean(),
            'judge_qa_acc': df['is_correct_judge_qa'].mean(),
            'verdict_acc': df['is_correct_verdict'].mean(),
            'debater_qa_n_correct': df['is_correct_debater_qa'].sum(),
            'judge_qa_n_correct': df['is_correct_judge_qa'].sum(),
            'verdict_n_correct': df['is_correct_verdict'].sum(),
            'n_total': n
        })
    results_df = pd.DataFrame(results)
    return results_df


def plot_gain_scatter(results_df, n_choices, over: Literal["gap", "judge_qa"] = 'gap'):

    fig, ax = plt.subplots(figsize=(10, 6))

    results_df[f'debater_minus_judge'] = results_df[f'debater_qa_acc'] - results_df[f'judge_qa_acc']

    if over == 'gap':
        xfield = f'debater_minus_judge'
    elif over == 'judge_qa':
        xfield = f'judge_qa_acc'

    x = results_df[xfield]
    y = results_df[f'verdict_minus_judge_qa']

    names, color_map = sort_and_color_by_model_family(results_df['name'].unique())
    results_df['sort_order'] = results_df['name'].map({name: i for i, name in enumerate(names)})
    results_df = results_df.sort_values('sort_order').reset_index(drop=True)

    for i, row in results_df.iterrows():
        x_val = row[xfield]
        y_val = row[f'verdict_minus_judge_qa']
        
        ax.scatter(x_val, y_val, color=color_map[row['name']], alpha=0.7, s=80, label=row['name'])
        ax.annotate(row['name'], (x_val, y_val), fontsize=7, ha='left', va='bottom', alpha=0.7)

    if xfield == f'judge_qa_acc':
        # Add the f'debater_qa_acc' as a vertical line
        ax.axvline(x=results_df[f'debater_qa_acc'].mean(), color='gray', linestyle='--', linewidth=2, alpha=0.5, label=f'Debater QA Accuracy')
        # Add the change line as a vertical line - it's 1/N_choices
        ax.axvline(x=1/n_choices, color='red', linestyle='--', linewidth=2, alpha=0.5, label='chance')

    # linear fit
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # r_squared = r_value**2
    # x_line = np.linspace(x.min(), x.max(), 100)
    # y_line = slope * x_line + intercept
    # ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label=f'Linear fit')
    # stats_text = f'R² = {r_squared:.3f}\np = {p_value:.4f}\nslope = {slope:.3f}'
    # ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            # verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    field_to_axis_label_map = {
        f'debater_minus_judge': 'Debater QA - Judge QA (Gap)',
        f'judge_qa_acc': 'Judge QA Accuracy'
    }

    ax.set_xlabel(field_to_axis_label_map[xfield], fontsize=12)
    ax.set_ylabel('Verdict - Judge QA (Gain)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='both', alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_delta_over_delta(merged, suffixes,xfield: Literal['gap_delta', 'judge_delta'], yfield: Literal['gain_delta', 'gap_delta'], n_min: int = 50):
    merged['gap_delta'] = merged[f'debater_minus_judge_qa{suffixes[0]}'] - merged[f'debater_minus_judge_qa{suffixes[1]}']
    merged['gain_delta'] = merged[f'verdict_minus_judge_qa{suffixes[0]}'] - merged[f'verdict_minus_judge_qa{suffixes[1]}']
    merged['judge_delta'] = merged[f'judge_qa_acc{suffixes[0]}'] - merged[f'judge_qa_acc{suffixes[1]}']

    fig, ax = plt.subplots(figsize=(12, 6))

    names, color_map = sort_and_color_by_model_family(merged['name'])
    for name in names:
        row = merged[merged['name'] == name].iloc[0]
        n_0 = int(row[f'n_total{suffixes[0]}'])
        n_1 = int(row[f'n_total{suffixes[1]}'])
        if n_0 < n_min or n_1 < n_min:
            print(f'skipping {name} because too few samples: n_0 = {n_0} and n_1 = {n_1}')
            # drop from the merged dataframe
            merged = merged[merged['name'] != name]
            continue
        ax.scatter(row[xfield], row[yfield], 
                color=color_map[name], 
                label=f"{name} (N={n_0}, {n_1})",
                alpha=0.7,
                s=100)

    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(merged[xfield], merged[yfield])
    x_line = np.linspace(merged[xfield].min(), merged[xfield].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=2)

    # ax.text(0.05, 0.95, f'$R^2$={r**2:.3f}\nslope={slope:.3f}\np={p:.3f}', 
    #         transform=ax.transAxes, fontsize=11, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(0.75, 0.95, f'$R^2$={r**2:.3f}\nslope={slope:.3f}\np={p:.3f}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fieldname_to_label_map = {
        'gap_delta': 'Gap Delta',
        'judge_delta': 'Judge Delta',
        'gain_delta': 'Gain Delta'
    }

    ax.set_xlabel(fieldname_to_label_map[xfield])
    ax.set_ylabel(fieldname_to_label_map[yfield])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=f'N={suffixes[0]}, {suffixes[1]}')

    # ax.set_xlim(-.5, .5)
    # ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.show()


def plot_gain_over_gap(results, xfield, yfield):

    xfield = 'gap'
    # xfield = 'judge_acc'
    yfield = 'gain'

    x = results[xfield].values.astype(float)
    y = results[yfield].values.astype(float)
    weights = results['count'].values.astype(float)

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y, sample_weight=weights)
    slope = model.coef_[0]
    intercept = model.intercept_

    y_pred = model.predict(x.reshape(-1, 1))
    ss_res = np.sum(weights * (y - y_pred)**2)
    ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
    r_squared = 1 - (ss_res / ss_tot)

    pearson_r, p_value = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(results[xfield], results[yfield], 
                        c=results['count'], s=100, cmap='Blues', alpha=0.7, edgecolors='black')

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.5, label=f'Weighted fit (slope={slope:.3f})')

    for i, row in results.iterrows():
        ax.annotate(row['category'], (row[xfield], row[yfield]), 
                    fontsize=8, ha='center', va='bottom', alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    stats_text = f'R² = {r_squared:.3f}\np = {p_value:.4f}\nslope = {slope:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fieldname_to_label_map = {
        'gap': 'Gap (Debater QA - Judge QA)',
        'gain': 'Gain (Verdict - Judge QA)',
        'judge_acc': 'Judge QA'
    }


    ax.set_xlabel(fieldname_to_label_map[xfield])
    ax.set_ylabel(fieldname_to_label_map[yfield])
    ax.set_title(f'Verdict: {verdict_run_id}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Samples')

    plt.tight_layout()
    plt.show()
