"""Select on validation, then evaluate four frozen models on a future holdout."""
import argparse
import json
import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import ALL_FEATURES, BASE_FEATURES


def split_by_time(frame, train_end_day=27, test_start_day=36):
    """Keep complete minutes in disjoint training, validation and test periods."""
    if not 0 < train_end_day < test_start_day:
        raise ValueError('Require 0 < train-end-day < test-start-day')
    train = frame.loc[frame.minute < train_end_day * 1440]
    validation = frame.loc[
        (frame.minute >= train_end_day * 1440) & (frame.minute < test_start_day * 1440)
    ]
    test = frame.loc[frame.minute >= test_start_day * 1440]
    for name, part in [('train', train), ('validation', validation), ('test', test)]:
        if part.empty or part.is_fraud.nunique() != 2:
            raise ValueError(f'{name} needs both fraud and legitimate transactions')
    return train, validation, test


def evaluate(labels, scores, amounts, top_k=100):
    """Measure ranking quality and alerted value; production uses a 100-alert budget."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if not (len(labels) == len(scores) == len(amounts)) or len(labels) == 0:
        raise ValueError('Labels, scores and amounts must have equal, nonzero lengths')
    if not np.isfinite(scores).all() or top_k <= 0:
        raise ValueError('Scores must be finite and the alert budget positive')
    # Input is ordered by minute and ID, making tied scores reproducible.
    selected = np.argsort(-scores, kind='stable')[:top_k]
    fraud_value = float(np.sum(amounts * labels))
    captured = float(np.sum(amounts[selected] * labels[selected]))
    return {
        'pr_auc': float(average_precision_score(labels, scores)),
        'roc_auc': float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        'alert_count': len(selected),
        'precision_at_100': float(labels[selected].mean()),
        'frauds_at_100': int(labels[selected].sum()),
        'fraud_value_at_100': captured,
        'total_fraud_value': fraud_value,
        'fraud_value_recall_at_100': captured / fraud_value if fraud_value else 0.0,
    }


def model_candidates(model_name, seed):
    if model_name == 'logistic':
        for regularization in (0.1, 1.0):
            estimator = LogisticRegression(
                C=regularization, max_iter=3000, class_weight='balanced',
                solver='lbfgs', random_state=seed,
            )
            yield {'C': regularization}, make_pipeline(
                SimpleImputer(strategy='median'), StandardScaler(), estimator
            )
    else:
        for depth in (3, 5):
            estimator = XGBClassifier(
                n_estimators=180, max_depth=depth, learning_rate=0.06,
                subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                n_jobs=4, eval_metric='logloss', random_state=seed,
            )
            yield {'max_depth': depth}, make_pipeline(
                SimpleImputer(strategy='median'), estimator
            )


def split_summary(frame):
    return {
        'rows': len(frame), 'fraud_rows': int(frame.is_fraud.sum()),
        'fraud_prevalence': float(frame.is_fraud.mean()),
        'start_minute': int(frame.minute.min()), 'end_minute': int(frame.minute.max()),
        'fraud_value': float((frame.amount * frame.is_fraud).sum()),
    }


def train_models(database_path, output_path, train_end_day=27, test_start_day=36, seed=42):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        frame = connection.execute(
            'SELECT * FROM model_features ORDER BY minute, transaction_id'
        ).df()
    if not np.isfinite(frame[ALL_FEATURES].to_numpy()).all():
        raise ValueError('Features contain nonfinite values')
    train, validation, test = split_by_time(frame, train_end_day, test_start_day)
    results = {
        'dataset': split_summary(frame),
        'splits': {name: split_summary(part) for name, part in
                   [('train', train), ('validation', validation), ('test', test)]},
        'feature_count': len(ALL_FEATURES),
        'selection_metric': 'validation average precision',
        'pr_auc_definition': 'average precision (not trapezoidal integration)',
        'value_definition': 'Fraud transaction amount among top 100; not recovered funds',
        'models': {},
    }
    scored = test[['transaction_id', 'minute', 'sender_id', 'receiver_id', 'amount', 'is_fraud']].copy()
    fitted_models = {}
    # Complete all validation choices before reading any holdout scores.
    for feature_set, columns in [('transaction_only', BASE_FEATURES), ('network_context', ALL_FEATURES)]:
        for model_name in ('logistic', 'xgboost'):
            key = f'{feature_set}_{model_name}'
            best_score = -1.0
            trials = []
            for parameters, pipeline in model_candidates(model_name, seed):
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    pipeline.fit(train[columns], train.is_fraud)
                validation_scores = pipeline.predict_proba(validation[columns])[:, 1]
                validation_ap = float(average_precision_score(validation.is_fraud, validation_scores))
                trials.append({'parameters': parameters, 'validation_pr_auc': validation_ap})
                if validation_ap > best_score:
                    best_score = validation_ap
                    best_pipeline = pipeline
                    best_parameters = parameters
            results['models'][key] = {
                'features': len(columns), 'selected_parameters': best_parameters,
                'validation_pr_auc': best_score, 'validation_trials': trials,
            }
            fitted_models[key] = (best_pipeline, columns)
            print(f'{key}: selected {best_parameters}; validation PR-AUC={best_score:.4f}', flush=True)

    for key, (pipeline, columns) in fitted_models.items():
        scores = pipeline.predict_proba(test[columns])[:, 1]
        results['models'][key]['test'] = evaluate(test.is_fraud, scores, test.amount)
        scored[key] = scores
        joblib.dump({
            'pipeline': pipeline, 'features': columns, 'training_end_minute': int(train.minute.max()),
            'validation_end_minute': int(validation.minute.max()),
            'test_start_minute': test_start_day * 1440,
        }, output_path / f'{key}.joblib')
    # Export after every score column exists so all alert files share one schema.
    for key in fitted_models:
        alerts = scored.sort_values(key, ascending=False, kind='stable').head(100)
        alerts.to_csv(output_path / f'{key}_top100.csv', index=False)
    scored.to_csv(output_path / 'scored_holdout.csv', index=False)
    (output_path / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/wallet.duckdb')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--train-end-day', type=int, default=27)
    parser.add_argument('--test-start-day', type=int, default=36)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_models(args.db, args.out, args.train_end_day, args.test_start_day, args.seed)
