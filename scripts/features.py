"""Build features using only transactions from strictly earlier minutes."""
import argparse
import json
from pathlib import Path

import duckdb

WINDOWS = {'15m': 15, '1h': 60, '6h': 360, '1d': 1440, '7d': 10080}
METRICS = {
    'count': ('count(*)', 'Number of prior transactions'),
    'total': ('sum(amount)', 'Total prior transaction amount'),
    'mean': ('avg(amount)', 'Mean prior transaction amount'),
    'max': ('max(amount)', 'Largest prior transaction amount'),
    'min': ('min(amount)', 'Smallest prior transaction amount'),
    'std': ('stddev_pop(amount)', 'Population standard deviation of prior amounts'),
    'distinct_partners': ('count(DISTINCT partner_id)', 'Unique prior counterparties'),
    'p2p_count': ("sum(CASE WHEN channel = 'P2P' THEN 1 ELSE 0 END)", 'Prior P2P count'),
    'qr_count': ("sum(CASE WHEN channel = 'QR' THEN 1 ELSE 0 END)", 'Prior QR count'),
    'large_count': ('sum(CASE WHEN amount > 3000 THEN 1 ELSE 0 END)', 'Prior amounts above 3000'),
    'small_count': ('sum(CASE WHEN amount < 300 THEN 1 ELSE 0 END)', 'Prior amounts below 300'),
}
BASE_FEATURES = ['amount', 'hour_of_day', 'weekday', 'is_p2p', 'is_qr']
HISTORY_FEATURES = [
    f'{role}_{window}_{metric}'
    for role in ('sender', 'receiver') for window in WINDOWS for metric in METRICS
]
ALL_FEATURES = BASE_FEATURES + HISTORY_FEATURES


def role_sql(role):
    """Describe one endpoint's prior activity without reading fraud labels."""
    identity = 'sender_id' if role == 'sender' else 'receiver_id'
    partner = 'receiver_id' if role == 'sender' else 'sender_id'
    expressions = ['transaction_id']
    for window_name, minutes in WINDOWS.items():
        # Minute-resolution input cannot reliably order events within a minute.
        frame = (f'PARTITION BY wallet_id ORDER BY minute '
                 f'RANGE BETWEEN {minutes} PRECEDING AND 1 PRECEDING')
        for metric_name, (expression, _) in METRICS.items():
            expressions.append(f'COALESCE({expression} OVER ({frame}), 0) '
                               f'AS {role}_{window_name}_{metric_name}')
    return f"""
        SELECT {', '.join(expressions)} FROM (
            SELECT transaction_id, {identity} AS wallet_id,
                {partner} AS partner_id, minute, amount, channel FROM transactions
        )
    """


def feature_sql():
    return f"""
        WITH sender AS ({role_sql('sender')}), receiver AS ({role_sql('receiver')})
        SELECT t.*, (t.minute % 1440) / 60.0 AS hour_of_day,
            t.minute // 1440 % 7 AS weekday,
            CAST(t.channel = 'P2P' AS INTEGER) AS is_p2p,
            CAST(t.channel = 'QR' AS INTEGER) AS is_qr,
            sender.* EXCLUDE(transaction_id), receiver.* EXCLUDE(transaction_id)
        FROM transactions t JOIN sender USING(transaction_id)
        JOIN receiver USING(transaction_id)
        ORDER BY t.minute, t.transaction_id
    """


def feature_dictionary():
    """Document the exact predictor names, meanings and availability rules."""
    descriptions = ['Current requested amount', 'Fractional hour in simulated day',
                    'Simulated weekday; day zero is Monday', 'Current channel is P2P',
                    'Current channel is QR; BANK is the reference channel']
    records = [dict(name=name, description=description, availability='At request time')
               for name, description in zip(BASE_FEATURES, descriptions)]
    for role in ('sender', 'receiver'):
        for window_name, minutes in WINDOWS.items():
            for metric, (_, description) in METRICS.items():
                records.append(dict(name=f'{role}_{window_name}_{metric}',
                                    description=f'{description} for this {role} in the {role} role',
                                    availability=f'[minute - {minutes}, minute - 1], inclusive; empty = 0'))
    return records


def build_features(database_path, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
        # Fixed aggregation order avoids tiny parallel-sum changes affecting model fitting.
        connection.execute('SET threads = 1')
        connection.execute('CREATE OR REPLACE TABLE model_features AS ' + feature_sql())
        row_count = connection.execute('SELECT count(*) FROM model_features').fetchone()[0]
    (output_path / 'feature_dictionary.json').write_text(
        json.dumps(feature_dictionary(), indent=2), encoding='utf-8')
    print(f'Built {len(ALL_FEATURES)} features ({len(HISTORY_FEATURES)} historical) for {row_count:,} rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/wallet.duckdb')
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()
    build_features(args.db, args.out)
