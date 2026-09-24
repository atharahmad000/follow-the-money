"""Audit stored features and independently reproduce saved-model evaluation."""
import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from features import ALL_FEATURES, HISTORY_FEATURES, WINDOWS
from train import evaluate


def historical_reference(transactions, transaction):
    """Slow, explicit reference calculation for auditing selected rows."""
    expected = {}
    for role in ('sender', 'receiver'):
        identity = f'{role}_id'
        partner = 'receiver_id' if role == 'sender' else 'sender_id'
        for window_name, minutes in WINDOWS.items():
            history = transactions.loc[
                (transactions[identity] == transaction[identity])
                & (transactions.minute >= transaction.minute - minutes)
                & (transactions.minute < transaction.minute)
            ]
            amounts = history.amount
            values = {
                'count': len(history), 'total': amounts.sum(),
                'mean': amounts.mean() if len(history) else 0,
                'max': amounts.max() if len(history) else 0,
                'min': amounts.min() if len(history) else 0,
                'std': amounts.std(ddof=0) if len(history) else 0,
                'distinct_partners': history[partner].nunique(),
                'p2p_count': int((history.channel == 'P2P').sum()),
                'qr_count': int((history.channel == 'QR').sum()),
                'large_count': int((amounts > 3000).sum()),
                'small_count': int((amounts < 300).sum()),
            }
            expected.update({f'{role}_{window_name}_{metric}': value
                             for metric, value in values.items()})
    return pd.Series(expected).reindex(HISTORY_FEATURES)


def verify_artifacts(database_path, output_path, minimum_rows=100000):
    """Check the data, sample historical calculations, and reproduce saved scores."""
    output_path = Path(output_path)
    results = json.loads((output_path / 'metrics.json').read_text(encoding='utf-8'))
    with duckdb.connect(str(database_path), read_only=True) as connection:
        transactions = connection.execute(
            'SELECT * FROM transactions ORDER BY minute, transaction_id'
        ).df()
        features = connection.execute(
            'SELECT * FROM model_features ORDER BY minute, transaction_id'
        ).df()
    assert len(transactions) >= minimum_rows, 'Insufficient transactions'
    assert len(transactions) == len(features) == results['dataset']['rows']
    assert len(ALL_FEATURES) >= 100 and len(set(ALL_FEATURES)) == len(ALL_FEATURES)
    assert not set(ALL_FEATURES) & {'is_fraud', 'transaction_id', 'sender_id', 'receiver_id', 'minute'}
    pd.testing.assert_frame_equal(features[transactions.columns], transactions)
    assert np.isfinite(features[ALL_FEATURES].to_numpy()).all()
    assert features[ALL_FEATURES].nunique().gt(1).all(), 'A feature is constant in this dataset'

    sample = features.sample(n=min(40, len(features)), random_state=17)
    fraud_sample = features.loc[features.is_fraud == 1].head(10)
    sample = pd.concat([sample, features.head(5), fraud_sample]).drop_duplicates('transaction_id')
    for _, row in sample.iterrows():
        expected = historical_reference(transactions, row)
        np.testing.assert_allclose(row[HISTORY_FEATURES].to_numpy(dtype=float),
                                   expected.to_numpy(dtype=float), rtol=1e-8, atol=1e-7)

    splits = results['splits']
    assert splits['train']['end_minute'] < splits['validation']['start_minute']
    assert splits['validation']['end_minute'] < splits['test']['start_minute']
    test = features.loc[features.minute >= splits['test']['start_minute']]
    scored = pd.read_csv(output_path / 'scored_holdout.csv')
    np.testing.assert_array_equal(scored.transaction_id, test.transaction_id)
    checked_models = []
    for key, model_result in results['models'].items():
        artifact = joblib.load(output_path / f'{key}.joblib')
        scores = artifact['pipeline'].predict_proba(test[artifact['features']])[:, 1]
        np.testing.assert_allclose(scores, scored[key], rtol=1e-7, atol=1e-9)
        reproduced = evaluate(test.is_fraud, scores, test.amount)
        for metric, value in reproduced.items():
            np.testing.assert_allclose(value, model_result['test'][metric])
        alerts = pd.read_csv(output_path / f'{key}_top100.csv')
        expected_alerts = scored.sort_values(key, ascending=False, kind='stable').head(100)
        np.testing.assert_array_equal(alerts.transaction_id, expected_alerts.transaction_id)
        assert model_result['validation_pr_auc'] == max(
            trial['validation_pr_auc'] for trial in model_result['validation_trials']
        )
        checked_models.append(key)

    audit = {
        'status': 'passed', 'rows': len(transactions), 'features': len(ALL_FEATURES),
        'nonconstant_features': int(features[ALL_FEATURES].nunique().gt(1).sum()),
        'reference_rows_checked': len(sample), 'historical_features_per_reference_row': len(HISTORY_FEATURES),
        'models_reloaded_and_metrics_reproduced': checked_models,
        'strictly_separated_time_splits': True,
    }
    (output_path / 'verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/wallet.duckdb')
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()
    verify_artifacts(args.db, args.out)
