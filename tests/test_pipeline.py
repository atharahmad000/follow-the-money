"""Behavioral tests for point-in-time features, loading and evaluation."""
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from features import ALL_FEATURES, HISTORY_FEATURES, feature_sql
from generate import generate
from load import load_transactions
from train import evaluate, split_by_time
from verify import historical_reference


def calculate_features(transactions):
    with duckdb.connect(':memory:') as connection:
        connection.register('transactions', transactions)
        return connection.execute(feature_sql()).df()


def example_transactions():
    # Deliberately hit the 15-minute boundary, same-minute ties and empty history.
    return pd.DataFrame({
        'transaction_id': range(9), 'minute': [0, 1, 14, 15, 15, 16, 60, 1440, 10081],
        'sender_id': [1, 1, 2, 1, 1, 2, 1, 1, 1],
        'receiver_id': [3, 3, 3, 3, 4, 3, 3, 3, 3],
        'amount': [100., 4000., 300., 70., 8000., 200., 3000., 900., 2000.],
        'channel': ['P2P', 'QR', 'BANK', 'QR', 'P2P', 'BANK', 'P2P', 'QR', 'P2P'],
        'is_fraud': [0, 0, 1, 1, 0, 0, 1, 0, 0],
    })


class PipelineTests(unittest.TestCase):
    def test_all_history_features_match_independent_reference(self):
        transactions = example_transactions()
        features = calculate_features(transactions)
        for _, row in features.iterrows():
            np.testing.assert_allclose(
                row[HISTORY_FEATURES].to_numpy(dtype=float),
                historical_reference(transactions, row).to_numpy(dtype=float),
                rtol=1e-9, atol=1e-7,
            )
        at_15 = features.loc[features.transaction_id == 3].iloc[0]
        self.assertEqual(at_15.sender_15m_count, 2)
        self.assertEqual(at_15.receiver_15m_count, 3)
        self.assertEqual(features.iloc[0].sender_7d_count, 0)

    def test_labels_and_future_changes_do_not_change_past_features(self):
        transactions = example_transactions()
        original = calculate_features(transactions)
        changed = transactions.copy()
        changed['is_fraud'] = 1 - changed.is_fraud
        changed.loc[changed.minute > 15, 'amount'] = 999999
        changed.loc[changed.minute > 15, 'receiver_id'] = 4
        recalculated = calculate_features(changed)
        pd.testing.assert_frame_equal(
            original.loc[original.minute <= 15, ALL_FEATURES],
            recalculated.loc[recalculated.minute <= 15, ALL_FEATURES],
        )

    def test_batch_features_equal_prefix_computation(self):
        transactions = example_transactions()
        full = calculate_features(transactions)
        prefix = calculate_features(transactions.loc[transactions.minute <= 15])
        pd.testing.assert_frame_equal(full.loc[full.minute <= 15].reset_index(drop=True), prefix)

    def test_current_and_same_minute_amounts_are_excluded(self):
        transactions = example_transactions()
        original = calculate_features(transactions)
        transactions.loc[transactions.minute == 15, 'amount'] = 999999
        changed = calculate_features(transactions)
        pd.testing.assert_frame_equal(
            original.loc[original.minute == 15, HISTORY_FEATURES],
            changed.loc[changed.minute == 15, HISTORY_FEATURES],
        )

    def test_minute_boundaries_keep_ties_together(self):
        frame = pd.DataFrame({
            'minute': [0, 1, 1439, 1439, 1440, 1440, 2879, 2880, 2880],
            'is_fraud': [0, 1, 0, 1, 0, 1, 0, 0, 1],
        })
        train, validation, test = split_by_time(frame, 1, 2)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 3)
        self.assertEqual(len(test), 2)
        with self.assertRaises(ValueError):
            split_by_time(frame, 2, 1)

    def test_alert_metrics_and_stable_ties(self):
        metrics = evaluate([1, 0, 1, 0], [0.9, 0.9, 0.5, 0.1], [10, 20, 90, 40], top_k=2)
        self.assertEqual(metrics['alert_count'], 2)
        self.assertEqual(metrics['precision_at_100'], 0.5)
        self.assertEqual(metrics['fraud_value_at_100'], 10)
        self.assertEqual(metrics['fraud_value_recall_at_100'], 0.1)
        self.assertAlmostEqual(metrics['pr_auc'], 7 / 12)
        self.assertEqual(evaluate([0, 1], [0, 1], [1, 2])['alert_count'], 2)

    def test_generator_is_reproducible_and_valid(self):
        first = generate(1000, 42)
        pd.testing.assert_frame_equal(first, generate(1000, 42))
        self.assertTrue((first.sender_id != first.receiver_id).all())
        self.assertTrue(first.minute.is_monotonic_increasing)
        self.assertTrue((first.amount > 0).all())

    def test_loader_rejects_invalid_input_without_replacing_table(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / 'transactions.csv'
            database_path = Path(folder) / 'wallet.duckdb'
            transactions = example_transactions()
            transactions.to_csv(csv_path, index=False)
            load_transactions(csv_path, database_path)
            for column, value in [('amount', -1), ('is_fraud', 2), ('channel', 'BAD'),
                                  ('transaction_id', 1), ('receiver_id', 1), ('minute', -1)]:
                invalid = transactions.copy()
                invalid.loc[0, column] = value
                invalid.to_csv(csv_path, index=False)
                with self.assertRaises(ValueError):
                    load_transactions(csv_path, database_path)
            with duckdb.connect(str(database_path), read_only=True) as connection:
                self.assertEqual(connection.execute('SELECT count(*) FROM transactions').fetchone()[0], 9)


if __name__ == '__main__':
    unittest.main()
