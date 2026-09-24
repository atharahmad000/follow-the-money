"""Validate a transaction CSV and load the analytical database."""
import argparse
from pathlib import Path

import duckdb


def load_transactions(csv_path, database_path):
    """Reject invalid input before replacing the persistent analytical tables."""
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("""
            CREATE OR REPLACE TEMP TABLE incoming AS
            SELECT * FROM read_csv(?, header=true, columns={
                'transaction_id': 'BIGINT', 'minute': 'BIGINT',
                'sender_id': 'BIGINT', 'receiver_id': 'BIGINT',
                'amount': 'DOUBLE', 'channel': 'VARCHAR', 'is_fraud': 'INTEGER'
            })
        """, [str(csv_path)])
        invalid_rows = connection.execute("""
            SELECT count(*) FROM incoming WHERE
                transaction_id IS NULL OR transaction_id < 0 OR
                minute IS NULL OR minute < 0 OR
                sender_id IS NULL OR sender_id < 0 OR
                receiver_id IS NULL OR receiver_id < 0 OR sender_id = receiver_id OR
                amount IS NULL OR NOT isfinite(amount) OR amount <= 0 OR
                channel IS NULL OR channel NOT IN ('P2P', 'QR', 'BANK') OR
                is_fraud IS NULL OR is_fraud NOT IN (0, 1)
        """).fetchone()[0]
        row_count, unique_ids = connection.execute(
            'SELECT count(*), count(DISTINCT transaction_id) FROM incoming'
        ).fetchone()
        if invalid_rows or row_count != unique_ids or row_count == 0:
            raise ValueError(f'Invalid input: {invalid_rows} invalid rows, '
                             f'{row_count - unique_ids} duplicate IDs, {row_count} rows')
        # Validate before replacing an existing usable fact table.
        connection.execute('BEGIN TRANSACTION')
        connection.execute('CREATE OR REPLACE TABLE transactions AS SELECT * FROM incoming')
        connection.execute('ALTER TABLE transactions ADD PRIMARY KEY (transaction_id)')
        connection.execute("""
            CREATE OR REPLACE VIEW daily_olap AS
            SELECT minute // 1440 AS day, channel, count(*) AS transactions,
                sum(amount) AS volume, sum(is_fraud) AS fraud_count,
                sum(amount * is_fraud) AS fraud_value
            FROM transactions GROUP BY ALL
        """)
        connection.execute('COMMIT')
    print(f'Loaded {row_count:,} validated transactions into {database_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/transactions.csv')
    parser.add_argument('--db', default='data/wallet.duckdb')
    args = parser.parse_args()
    load_transactions(args.csv, args.db)
