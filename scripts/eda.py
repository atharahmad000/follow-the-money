"""Query DuckDB from Python and export training-period exploratory analysis."""
import argparse
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_eda(database_path, output_path, train_end_day=27):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        summary = connection.execute("""
            SELECT count(*) AS rows, sum(is_fraud) AS fraud_rows,
                avg(is_fraud) AS fraud_prevalence, sum(amount * is_fraud) AS fraud_value,
                count(DISTINCT sender_id) AS senders, count(DISTINCT receiver_id) AS receivers
            FROM transactions
        """).df().iloc[0].to_dict()
        daily = connection.execute(
            'SELECT * FROM daily_olap WHERE day < ? ORDER BY day, channel',
            [train_end_day],
        ).df()
        amounts = connection.execute("""
            SELECT is_fraud, count(*) AS rows, min(amount) AS minimum,
                quantile_cont(amount, 0.5) AS median, quantile_cont(amount, 0.95) AS p95,
                max(amount) AS maximum, avg(amount) AS mean
            FROM transactions WHERE minute < ? GROUP BY is_fraud ORDER BY is_fraud
        """, [train_end_day * 1440]).df()
        channels = connection.execute("""
            SELECT channel, count(*) AS rows, avg(is_fraud) AS fraud_prevalence,
                sum(amount) AS volume FROM transactions WHERE minute < ?
            GROUP BY channel ORDER BY channel
        """, [train_end_day * 1440]).df()
    summary['eda_scope'] = f'Distribution analysis uses days 0 through {train_end_day - 1} only'
    for count_field in ('rows', 'fraud_rows', 'senders', 'receivers'):
        summary[count_field] = int(summary[count_field])
    summary['training_amounts'] = amounts.to_dict(orient='records')
    summary['training_channels'] = channels.to_dict(orient='records')
    (output_path / 'eda.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    daily.to_csv(output_path / 'eda_daily_training.csv', index=False)
    totals = daily.groupby('day')[['transactions', 'fraud_count']].sum()
    totals.plot(subplots=True, figsize=(10, 6),
                title=['Training transactions by day', 'Training simulated fraud by day'])
    plt.tight_layout()
    plt.savefig(output_path / 'eda.png', dpi=150)
    plt.close()
    print(f"EDA: {int(summary['rows']):,} total rows; "
          f"{summary['fraud_prevalence']:.3%} simulated fraud")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/wallet.duckdb')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--train-end-day', type=int, default=27)
    args = parser.parse_args()
    run_eda(args.db, args.out, args.train_end_day)
