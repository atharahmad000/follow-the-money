"""Create portfolio figures and a compact, auditable snapshot from a completed run."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from train import evaluate

ROOT = Path(__file__).resolve().parents[1]
MODEL_LABELS = {
    'transaction_only_logistic': 'Transaction-only / logistic',
    'transaction_only_xgboost': 'Transaction-only / XGBoost',
    'network_context_logistic': 'Historical network / logistic',
    'network_context_xgboost': 'Historical network / XGBoost',
}
COLORS = ['#64748b', '#9aa9ba', '#157c80', '#d08c25']
INK = '#15283c'
MUTED = '#536579'
PAPER = '#f7f9fc'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style_axes(axis):
    axis.set_facecolor(PAPER)
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.spines['bottom'].set_color('#cbd5e1')
    axis.tick_params(colors=MUTED, length=0)
    axis.grid(axis='x', color='#dde4ed', linewidth=0.7)
    axis.set_axisbelow(True)


def save_figure(figure, path):
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_model_comparison(metrics, destination):
    figure, axes = plt.subplots(1, 3, figsize=(14, 6.2), sharey=True)
    figure.patch.set_facecolor(PAPER)
    figure.subplots_adjust(left=0.25, right=0.97, top=0.74, bottom=0.23, wspace=0.19)
    figure.text(0.04, 0.91, 'Better ranking. A different value tradeoff.', fontsize=23,
                weight='bold', color=INK)
    figure.text(0.04, 0.84, 'Future-period evaluation | 100 alerts across the entire holdout',
                fontsize=12, color=MUTED)
    specifications = [
        ('pr_auc', 'PR-AUC', False),
        ('precision_at_100', 'Precision @100', True),
        ('fraud_value_recall_at_100', 'Fraud value share @100', True),
    ]
    for axis, (metric, title, percentage) in zip(axes, specifications):
        values = [metrics['models'][key]['test'][metric] for key in MODEL_LABELS]
        axis.barh(range(4), values, color=COLORS, height=0.5)
        axis.set_xlim(0, 1.22)
        axis.set_xticks([0, 0.5, 1])
        axis.set_title(title, loc='left', pad=18, fontsize=12, color=INK, weight='bold')
        axis.set_yticks(range(4), list(MODEL_LABELS.values()), fontsize=11)
        if percentage:
            axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        for position, value in enumerate(values):
            label = f'{value:.1%}' if percentage else f'{value:.3f}'
            axis.text(value + 0.035, position, label, va='center', fontsize=11, color=INK)
        style_axes(axis)
    axes[0].invert_yaxis()
    figure.text(0.04, 0.10,
                'Synthetic data only. PR-AUC = average precision. Value = labeled transaction amount, not recovered money.',
                fontsize=10, color=MUTED)
    figure.text(0.04, 0.055, 'Source: saved future-holdout metrics; the same alert budget is used for every model.',
                fontsize=10, color=MUTED)
    save_figure(figure, destination / 'model-comparison.png')


def plot_precision_recall(scores, metrics, destination):
    figure, axis = plt.subplots(figsize=(11, 6.8))
    figure.patch.set_facecolor(PAPER)
    figure.subplots_adjust(left=0.09, right=0.96, top=0.80, bottom=0.17)
    figure.text(0.09, 0.92, 'How precision changes as recall increases', fontsize=21,
                weight='bold', color=INK)
    figure.text(0.09, 0.86, 'Four models, one future holdout | curves computed from saved transaction scores',
                fontsize=11, color=MUTED)
    for (key, label), color in zip(MODEL_LABELS.items(), COLORS):
        precision, recall, _ = precision_recall_curve(scores.is_fraud, scores[key])
        average_precision = metrics['models'][key]['test']['pr_auc']
        axis.step(recall, precision, where='post', color=color, linewidth=1.8,
                  label=f'{label} (AP {average_precision:.3f})')
    prevalence = float(scores.is_fraud.mean())
    axis.axhline(prevalence, color=MUTED, linestyle='--', linewidth=1,
                 label=f'Holdout prevalence ({prevalence:.2%})')
    axis.set(xlim=(0, 1), ylim=(0, 1.04), xlabel='Recall: share of fraud transactions identified',
             ylabel='Precision: share of alerts that are fraud')
    axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    style_axes(axis)
    axis.legend(loc='center', bbox_to_anchor=(0.48, 0.49), fontsize=9,
                frameon=True, facecolor='white', edgecolor='#dde4ed')
    figure.text(0.09, 0.06, 'Simulated data. Empirical step curves from saved scores; AP denotes average precision.',
                fontsize=10, color=MUTED)
    save_figure(figure, destination / 'precision-recall.png')


def plot_time_safety(metrics, destination):
    figure, axes = plt.subplots(2, 1, figsize=(13, 6.5))
    figure.patch.set_facecolor(PAPER)
    figure.subplots_adjust(left=0.065, right=0.965, top=0.77, bottom=0.17, hspace=0.8)
    figure.text(0.065, 0.92, 'A prediction can only use the past', fontsize=23, weight='bold', color=INK)
    figure.text(0.065, 0.85, 'Two separate safeguards: chronological evaluation and strictly prior-minute features',
                fontsize=11, color=MUTED)
    axis = axes[0]
    axis.set_title('01  Evaluation periods', loc='left', fontsize=12, color=INK)
    for name, label, color in [('train', 'TRAIN', COLORS[0]),
                               ('validation', 'VALIDATION', COLORS[2]),
                               ('test', 'FUTURE HOLDOUT', COLORS[3])]:
        part = metrics['splits'][name]
        start = part['start_minute'] / 1440
        end = (part['end_minute'] + 1) / 1440
        axis.add_patch(Rectangle((start, 0.15), end - start, 0.65, facecolor=color))
        axis.text((start + end) / 2, 0.475, f'{label}\n{part["rows"]:,} rows',
                  ha='center', va='center', fontsize=11, color='white', weight='bold')
    boundaries = [metrics['splits'][name]['start_minute'] / 1440
                  for name in ('train', 'validation', 'test')]
    boundaries.append((metrics['splits']['test']['end_minute'] + 1) / 1440)
    axis.set(xlim=(boundaries[0], boundaries[-1]), ylim=(0, 1), yticks=[],
             xticks=boundaries, xlabel='Simulation day; zero-based')
    style_axes(axis)
    axis = axes[1]
    axis.set_title('02  Feature availability for a transaction at minute t', loc='left', fontsize=12, color=INK)
    for start, width, color, label in [
        (0, 0.65, COLORS[2], 'PRIOR EVENTS\nIncluded in the lookback window'),
        (0.67, 0.13, COLORS[3], 'MINUTE t\nExcluded'),
        (0.82, 0.18, '#dbe3ec', 'FUTURE\nExcluded'),
    ]:
        axis.add_patch(Rectangle((start, 0.1), width, 0.75, facecolor=color))
        axis.text(start + width / 2, 0.475, label, ha='center', va='center', fontsize=10,
                  color=INK if start > 0.8 else 'white', weight='bold')
    axis.set(xlim=(0, 1), ylim=(0, 1), yticks=[], xticks=[0, 0.65, 0.735, 1],
             xticklabels=['t - window', 't - 1', 't', 'Later'])
    style_axes(axis)
    figure.text(0.065, 0.065, 'History panel is schematic, not to scale. Same-minute peers and fraud labels never enter historical features.',
                fontsize=10, color=MUTED)
    save_figure(figure, destination / 'time-safety.png')


def plot_training_activity(results_path, destination):
    daily = pd.read_csv(results_path / 'eda_daily_training.csv')
    totals = daily.groupby('day')[['transactions', 'fraud_count']].sum()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    figure.patch.set_facecolor(PAPER)
    figure.subplots_adjust(left=0.075, right=0.96, top=0.73, bottom=0.24, wspace=0.27)
    figure.text(0.075, 0.9, 'Understand the simulation before modeling', fontsize=21, weight='bold', color=INK)
    for axis, column, title, color in zip(
        axes, ['transactions', 'fraud_count'],
        ['All training transactions', 'Planted fraud in the training period'], [COLORS[2], COLORS[3]]
    ):
        axis.plot(totals.index, totals[column], color=color, linewidth=2.2)
        axis.fill_between(totals.index, totals[column], alpha=0.10, color=color)
        axis.set(title=title, xlabel='Simulation day', ylabel='Transactions', ylim=(0, None))
        style_axes(axis)
    figure.text(0.075, 0.08, 'Training period only. Fraud starts on day 14 by construction; this pattern is a simulation assumption.',
                fontsize=10, color=MUTED)
    save_figure(figure, destination / 'training-activity.png')


def export_snapshot(results_path, snapshot_path):
    snapshot_path.mkdir(parents=True, exist_ok=True)
    for filename in ['metrics.json', 'verification.json', 'eda.json', 'feature_dictionary.json']:
        shutil.copyfile(results_path / filename, snapshot_path / filename)
    manifest = read_json(results_path / 'run_manifest.json')
    # Keep reproducibility evidence without publishing machine-specific home paths.
    metadata = {key: manifest[key] for key in
                ['status', 'seed', 'input_source', 'packages', 'input_sha256', 'seconds', 'source_sha256']}
    metadata['python'] = manifest['python'].split()[0]
    metadata['visualization_script_sha256'] = sha256(Path(__file__))
    metadata['commands'] = [
        {'argv': ['python', *[argument.replace(str(ROOT), '<project>').replace('\\', '/')
                              for argument in stage['argv'][1:]]],
         'exit_code': stage['exit_code'], 'seconds': stage['seconds']}
        for stage in manifest['commands']
    ]
    metadata['chart_sources_sha256'] = {
        filename: sha256(results_path / filename)
        for filename in ['metrics.json', 'scored_holdout.csv', 'eda_daily_training.csv']
    }
    (snapshot_path / 'run-summary.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/assets')
    parser.add_argument('--snapshot-dir', type=Path, default=ROOT / 'docs/results')
    args = parser.parse_args()
    manifest = read_json(args.results / 'run_manifest.json')
    audit = read_json(args.results / 'verification.json')
    if manifest['status'] != 'passed' or audit['status'] != 'passed':
        raise ValueError('Publish figures only from a completed, verified run')
    metrics = read_json(args.results / 'metrics.json')
    scores = pd.read_csv(args.results / 'scored_holdout.csv')
    for key in MODEL_LABELS:
        reproduced = evaluate(scores.is_fraud, scores[key], scores.amount)
        for metric, value in reproduced.items():
            np.testing.assert_allclose(value, metrics['models'][key]['test'][metric],
                                       rtol=1e-10, atol=1e-10)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelcolor': INK})
    plot_model_comparison(metrics, args.out)
    plot_precision_recall(scores, metrics, args.out)
    plot_time_safety(metrics, args.out)
    plot_training_activity(args.results, args.out)
    export_snapshot(args.results, args.snapshot_dir)
    print(f'Wrote four figures to {args.out} and verified snapshots to {args.snapshot_dir}')


if __name__ == '__main__':
    main()
