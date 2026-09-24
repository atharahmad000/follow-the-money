"""Generate simulated wallet transactions; these are not provider data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SIMULATION_DAYS = 45


def add_scam_bursts(rng, minutes, senders, receivers, amounts, labels):
    mule_ids = np.arange(4800, 4850)
    for day in range(14, SIMULATION_DAYS):
        for mule in rng.choice(mule_ids, size=3, replace=False):
            start_minute = day * 1440 + int(rng.integers(100, 1150))
            available = np.flatnonzero(
                (minutes >= start_minute) & (minutes < start_minute + 90) & (labels == 0)
            )
            if len(available) < 10:
                continue
            selected = rng.choice(available, size=9, replace=False)
            senders[selected] = rng.choice(4500, size=len(selected), replace=False)
            receivers[selected] = mule
            amounts[selected] = np.round(rng.lognormal(6.7, 0.8, len(selected)), 2)
            labels[selected] = 1

    # A final collector transfer aggregates earlier scam receipts for each mule.
    # This is a simplified simulation, not a balance-conserving wallet ledger.
    for mule in mule_ids:
        incoming = np.flatnonzero((receivers == mule) & (labels == 1))
        if len(incoming) < 3:
            continue
        last_receipt = minutes[incoming[-1]]
        available = np.flatnonzero(
            (minutes > last_receipt) & (minutes <= last_receipt + 120) & (labels == 0)
        )
        if len(available):
            selected = int(available[0])
            senders[selected] = mule
            receivers[selected] = 4950 + int(mule % 10)
            amounts[selected] = round(float(amounts[incoming].sum() * 0.75), 2)
            labels[selected] = 1


def generate(rows=120_000, seed=42):
    """Return a repeatable 45-day sample of payments with planted fraud labels."""
    if rows < 1000:
        raise ValueError('At least 1,000 rows required for the generator')
    rng = np.random.default_rng(seed)
    minutes = np.sort(rng.integers(0, 1440 * SIMULATION_DAYS, size=rows))
    senders = rng.integers(0, 5000, size=rows)
    receivers = rng.integers(0, 5000, size=rows)
    amounts = np.round(rng.lognormal(6.1, 1.05, rows), 2)
    labels = np.zeros(rows, dtype=int)
    add_scam_bursts(rng, minutes, senders, receivers, amounts, labels)

    # Busy legitimate merchants provide a comparison with concentrated receipts.
    merchant_payments = rng.choice(np.flatnonzero(labels == 0), size=rows // 35, replace=False)
    receivers[merchant_payments] = rng.choice(np.arange(4900, 4950), size=len(merchant_payments))
    same_wallet = senders == receivers
    senders[same_wallet] = (senders[same_wallet] + 1) % 5000
    return pd.DataFrame({
        'transaction_id': np.arange(rows), 'minute': minutes,
        'sender_id': senders, 'receiver_id': receivers, 'amount': amounts,
        'channel': rng.choice(['P2P', 'QR', 'BANK'], rows, p=[0.7, 0.2, 0.1]),
        'is_fraud': labels,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=120000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='data/transactions.csv')
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transactions = generate(args.rows, args.seed)
    transactions.to_csv(output_path, index=False)
    print(f'{len(transactions):,} simulated rows; {transactions.is_fraud.sum():,} fraud rows')
