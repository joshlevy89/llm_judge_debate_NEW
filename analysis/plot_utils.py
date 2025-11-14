import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
from typing import Literal
from typing import Literal
from scipy import stats
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.proportion import proportions_ztest
import numpy as np


def plot_accuracy_bars(results_df, ax=None, color_map=None, show_sig=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 6))
    
    x = np.arange(len(results_df))
    width = 0.25

    bars1 = ax.bar(x - width, results_df['debater_qa_acc'], width, label='Debater QA', alpha=0.8)
    bars2 = ax.bar(x, results_df['judge_qa_acc'], width, label='Judge QA', alpha=0.8)
    bars3 = ax.bar(x + width, results_df['verdict_acc'], width, label='Verdict', alpha=0.8)

    for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
        ratio1 = round(results_df.iloc[i]['debater_qa_acc'], 2)
        ratio2 = round(results_df.iloc[i]['judge_qa_acc'], 2)
        ratio3 = round(results_df.iloc[i]['verdict_acc'], 2)
        
        n_total = results_df.iloc[i]['n_total']
        judge_qa_correct = results_df.iloc[i]['judge_qa_n_correct']
        debater_qa_correct = results_df.iloc[i]['debater_qa_n_correct']
        verdict_correct = results_df.iloc[i]['verdict_n_correct']


        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02, 
                f"{ratio1}\n{debater_qa_correct:.0f}/{n_total}", 
                ha='center', va='bottom', fontsize=9)
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02, 
                f"{ratio2}\n{judge_qa_correct:.0f}/{n_total}", 
                ha='center', va='bottom', fontsize=9)
        ax.text(b3.get_x() + b3.get_width()/2, b3.get_height() + 0.02, 
                f"{ratio3}\n{verdict_correct:.0f}/{n_total}", 
                ha='center', va='bottom', fontsize=9)

        # Test significance of gain (judge QA vs verdict)
        z_stat, p_value = test_gain_significance(judge_qa_correct, n_total, verdict_correct, n_total)

        # Add significance bracket if p < 0.05
        if p_value < 0.05 and show_sig:
            # Get bar positions and heights
            b2_x = b2.get_x() + b2.get_width()/2
            b3_x = b3.get_x() + b3.get_width()/2
            max_height = max(b2.get_height(), b3.get_height())

            # Draw horizontal line connecting the bars
            bracket_y = max_height + 0.30  # Position above the taller bar and text labels
            ax.plot([b2_x, b3_x], [bracket_y, bracket_y], 'k-', linewidth=1.5)

            # Add vertical ticks at ends
            ax.plot([b2_x, b2_x], [max_height + 0.20, bracket_y], 'k-', linewidth=1.5)
            ax.plot([b3_x, b3_x], [max_height + 0.20, bracket_y], 'k-', linewidth=1.5)

            # Add p-value text
            if p_value < 0.001:
                sig_text = '***'
            elif p_value < 0.01:
                sig_text = '**'
            elif p_value < 0.05:
                sig_text = '*'
            else:
                sig_text = f'p={p_value:.3f}'

            ax.text((b2_x + b3_x)/2, bracket_y + 0.01, sig_text,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Accuracy')
    # Adjust ylim to accommodate significance brackets
    current_ylim = ax.get_ylim()
    ax.set_ylim(0, max(current_ylim[1], 1.25))  # Ensure space for brackets above text
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(x)
    return ax, plt


def test_gain_significance(judge_correct, judge_total, verdict_correct, verdict_total, alternative='two-sided'):
    """
    Test if the gain (verdict_acc - judge_qa_acc) is statistically significant.
    """
    successes = np.array([verdict_correct, judge_correct])
    totals = np.array([verdict_total, judge_total])
    z_stat, p_value = proportions_ztest(successes, totals, alternative=alternative)

    return z_stat, p_value


def plot_accuracy_bars_single(results_df, ax=None, show_sig=False):
    ax, plt = plot_accuracy_bars(results_df, ax=ax, show_sig=show_sig)
    ax.set_xticklabels(results_df['name'], rotation=0, ha='center')
    ax.figure.set_size_inches(5, 3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return ax, plt


def plot_verdict_difference(results_df, ax=None, type='gain'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 2))
    
    if type == 'gain':
        bars =ax.bar(results_df['name'], results_df['gain'], color='purple')
        ylabel = 'Gain'
    elif type == 'pgr':
        bars = ax.bar(results_df['name'], results_df['pgr'], color='cyan')
        ylabel = 'PGR'
    elif type == 'gap':
        bars = ax.bar(results_df['name'], results_df['gap'], color='pink')
        ylabel = 'Gap'
    # ax.set_ylim(-.1, 0.2)
    # add a horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Judge Model')
    ax.set_xticklabels(results_df['name'], rotation=45, ha='right')
    return ax, bars


def plot_results_by_name(results_df, field='config_judge_model_verdicts'):
    if field == 'config_judge_model_verdicts':
        sorted_names, _ = sort_and_color_by_model_family(results_df['name'].unique())
        results_df = results_df.set_index('name').loc[sorted_names].reset_index()
    fig, ax = plt.subplots(3, 1, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
    ax_acc, _ = plot_accuracy_bars(results_df, ax=ax[0])
    ax_acc.set_xticklabels(results_df['name'], rotation=45, ha='right')
    ax_gain, _ = plot_verdict_difference(results_df, ax=ax[1])
    ax_gap, _ = plot_verdict_difference(results_df, ax=ax[2], type='gap')
    plt.tight_layout()
    return plt, ax_gain, ax_gap


def plot_gain_scatter(results_df, n_choices, over: Literal["gap", "judge_qa"] = 'gap'):

    fig, ax = plt.subplots(figsize=(10, 6))

    results_df[f'debater_minus_judge'] = results_df[f'debater_qa_acc'] - results_df[f'judge_qa_acc']

    if over == 'gap':
        xfield = f'debater_minus_judge'
    elif over == 'judge_qa':
        xfield = f'judge_qa_acc'

    x = results_df[xfield]
    y = results_df[f'gain']

    names, color_map = sort_and_color_by_model_family(results_df['name'].unique())
    results_df['sort_order'] = results_df['name'].map({name: i for i, name in enumerate(names)})
    results_df = results_df.sort_values('sort_order').reset_index(drop=True)

    for i, row in results_df.iterrows():
        x_val = row[xfield]
        y_val = row[f'gain']
        
        ax.scatter(x_val, y_val, color=color_map[row['name']], alpha=0.7, s=80, label=f"{row['name']} (N={row['n_total']})")
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



def plot_delta_over_delta(merged, suffixes,xfield: Literal['gap_delta', 'judge_delta'], yfield: Literal['gain_delta', 'gap_delta']):
    merged = merged.copy()
    merged['gap_delta'] = merged[f'gap{suffixes[0]}'] - merged[f'gap{suffixes[1]}']
    merged['gain_delta'] = merged[f'gain{suffixes[0]}'] - merged[f'gain{suffixes[1]}']
    merged['judge_delta'] = merged[f'judge_qa_acc{suffixes[0]}'] - merged[f'judge_qa_acc{suffixes[1]}']

    fig, ax = plt.subplots(figsize=(12, 6))

    names, color_map = sort_and_color_by_model_family(merged['name'])
    for name in names:
        row = merged[merged['name'] == name].iloc[0]
        n_0 = int(row[f'n_total{suffixes[0]}'])
        n_1 = int(row[f'n_total{suffixes[1]}'])
        ax.scatter(row[xfield], row[yfield], 
                color=color_map[name], 
                label=f"{name} (N={n_0}, {n_1})",
                alpha=0.7,
                s=100)
        # Add labels to each scatter dot
        ax.annotate(name, (row[xfield], row[yfield]), fontsize=9, ha='center', va='bottom', alpha=0.75)

    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(merged[xfield], merged[yfield])
    x_line = np.linspace(merged[xfield].min(), merged[xfield].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=2)

    # ax.text(0.05, 0.95, f'$R^2$={r**2:.3f}\nslope={slope:.3f}\np={p:.3f}', 
    #         transform=ax.transAxes, fontsize=11, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(0.05, 0.95, f'$R^2$={r**2:.3f}\nslope={slope:.3f}\np={p:.3f}', 
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
    weights = results['n_total'].values.astype(float)

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
                        c=results['n_total'], s=100, cmap='Blues', alpha=0.7, edgecolors='black')

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.5, label=f'Weighted fit (slope={slope:.3f})')

    for i, row in results.iterrows():
        ax.annotate(row['name'], (row[xfield], row[yfield]), 
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
    ax.set_title(f'Gain vs Gap by Category')
    ax.grid(True, alpha=0.3)
    ax.legend()

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Samples')

    plt.tight_layout()
    plt.show()



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




def plot_spaghetti(merged, suffixes):
    sorted_names, color_map = sort_and_color_by_model_family(merged['name'].unique())
    merged['sort_order'] = merged['name'].map({name: i for i, name in enumerate(sorted_names)})
    merged = merged.sort_values('sort_order').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in merged.iterrows():
        n_2choice = int(row[f'n_total{suffixes[0]}'])
        n_4choice = int(row[f'n_total{suffixes[1]}'])
        ax.plot([0, 1], [row[f'gain{suffixes[0]}'], row[f'gain{suffixes[1]}']], 
                marker='o', label=f"{row['name']} (N={n_2choice}, {n_4choice})", color=color_map[row['name']], linewidth=2, markersize=8, linestyle='--')

    ax.plot([0, 1], [merged[f'gain{suffixes[0]}'].median(), merged[f'gain{suffixes[1]}'].median()], marker='o', label='Median', color='black', linewidth=2, markersize=8, linestyle='-')
    ax.plot([0, 1], [merged[f'gain{suffixes[0]}'].mean(), merged[f'gain{suffixes[1]}'].mean()], marker='o', label='Mean', color='green', linewidth=2, markersize=8, linestyle='-')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(suffixes, fontsize=12)
    ax.set_ylabel('Gain', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_correctness_grid(temp_df, value_field='is_correct_verdict'):

    pivot_df = temp_df.pivot_table(
        index='config_judge_model_verdicts',
        columns='question_idx_debates',
        values=value_field,
        aggfunc='first'
    )

    pivot_df = pivot_df.astype(float)

    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(20, max(8, len(pivot_df) * 0.5)))

    data = pivot_df.fillna(0.5).values.astype(float)

    cmap = ListedColormap(['red', 'gray', 'green'])
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks(range(len(pivot_df)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=90, fontsize=8)
    ax.set_xticks([x - 0.5 for x in range(len(pivot_df.columns) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(pivot_df) + 1)], minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Question Index')
    ax.set_ylabel('Model')
    ax.set_title('Correctness Grid: Green=Correct, Red=Wrong, Gray=Missing')

    plt.tight_layout()
    plt.show()


def cdf(data, labels=None, ax=None, xlim_percentiles=None, xlim_ranges=None, xlim_window_from_median=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if not labels:
        labels = [str(i) for i in range(len(data))]

    for idx, d in enumerate(data):
        d = np.sort(d.dropna().copy())
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), label=f"{labels[idx]} (n={len(d)})")
        # plot dash line at .5 on y axis
        ax.axhline(y=0.5, color='gray', linestyle='--')

        
    # d1 = np.sort(d1.dropna().copy())
    # d2 = np.sort(d2.dropna().copy())
    # ax.plot(d1, np.arange(1, len(d1) + 1) / len(d1), label=f'Success (n={len(d1)})')
    # ax.plot(d2, np.arange(1, len(d2) + 1) / len(d2), label=f'Failure (n={len(d2)})')
    ax.set_ylabel('Empirical CDF')
    ax.grid(True, alpha=0.3)
    
    if xlim_percentiles:
        combined = np.concatenate(data)
        x1, x2 = np.percentile(combined, xlim_percentiles[0]), np.percentile(combined, xlim_percentiles[1])
        ax.set_xlim(x1, x2)
    elif xlim_ranges:
        ax.set_xlim(xlim_ranges[0], xlim_ranges[1])
    elif xlim_window_from_median:
        combined = np.concatenate(data)
        combined_median = np.percentile(combined, 50)
        ax.set_xlim(combined_median + xlim_window_from_median[0], combined_median + xlim_window_from_median[1])
    
    return ax