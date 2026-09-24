# Verification record

**Status: passed.** Verified on 2026-09-24 with Python 3.12.10 on Windows x64.

The final analytical pipeline was executed in two separately installed, pinned environments. The second execution generated its own CSV, database, models, and reports in previously absent directories. Both completed successfully and produced exactly matching model metrics, holdout scores, and top-100 alert tables.

## Evidence at a glance

| Check | Result |
| --- | --- |
| Transactions loaded and featured | 120,000 |
| Fraud rows / prevalence | 881 / 0.7341667% |
| Predictors | 115; all nonconstant in this simulation |
| Train / validation / holdout rows | 71,760 / 24,334 / 23,906 |
| Holdout fraud rows / prevalence | 267 / 1.1168744% |
| Behavioral tests | Eight passed in each environment |
| Independent feature calculation | 55 rows x 110 historical features matched |
| Saved-model audit | All four models reloaded; scores, metrics and alert IDs reproduced |
| Alert export schema | All four alert tables contain the same four model score columns |
| Default execution | Passed in 99.76 seconds |
| Separate-environment execution | Passed in 99.25 seconds |
| Reproduction comparison | Exact matching metrics, scores, alert tables, input and source hashes, and package versions |

Timing excludes dependency installation and chart generation. Both transaction CSVs have SHA-256:

```text
251da2c185651cb7d22258e9af98e0cc7db02e31894d6727d1cacc3adcd4892b
```

## Tracked evidence

These small files are included with the project, so they can be reviewed on GitHub without downloading model binaries or generating the data:

- [metrics.json](docs/results/metrics.json): split statistics, every validation trial and full-precision holdout metrics.
- [verification.json](docs/results/verification.json): independent feature and saved-artifact audit.
- [run-summary.json](docs/results/run-summary.json): package versions, source/input hashes, relative commands and timings.
- [reproduction.json](docs/results/reproduction.json): cross-environment equality checks.
- [feature_dictionary.json](docs/results/feature_dictionary.json): definitions and timing rules for all 115 predictors.
- [eda.json](docs/results/eda.json): SQL-derived exploratory summaries.

The local `outputs/pipeline.log` and `outputs/run_manifest.json` contain the complete original execution output and absolute command arguments. Equivalent files are in `polished_run/outputs`. These large/local artifacts are intentionally Git-ignored; the public run summary replaces machine-specific paths with `<project>`.

Source hashes describe bytes at analytical-run time. Figure layout was refined afterward; `visualization_script_sha256` in the public run summary identifies the final chart renderer. This presentation change does not affect features, model fitting, or evaluation. Line-ending normalization on another platform can change source-file hashes.

## Commands used

Setup commands previously executed from the project root:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip freeze | Set-Content -Encoding ascii requirements-lock.txt
python -m venv .venv-clean
.venv-clean/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m pip check
.venv-clean/Scripts/python.exe -m pip check
```

Both dependency checks reported no broken requirements. For a new installation, follow the README and install the pinned lock file directly in either environment.

Final analytical runs and publication export:

```powershell
.venv/Scripts/python.exe run.py --rows 120000 --seed 42
.venv-clean/Scripts/python.exe run.py --rows 120000 --seed 42 --data-dir polished_run/data --out polished_run/outputs
.venv/Scripts/python.exe scripts/visualize.py --results outputs --out docs/assets --snapshot-dir docs/results
```

The runner executes all eight tests before generating data and independently audits saved artifacts after training. The visualization script recomputes model metrics from saved score rows before producing the publication figures.

<details>
<summary><strong>Exact cross-environment comparison command</strong></summary>

```powershell
@'
import json
from pathlib import Path
import pandas as pd
first = Path('outputs')
second = Path('polished_run/outputs')
def read(folder, name):
    return json.loads((folder / name).read_text(encoding='utf-8'))
a, b = read(first, 'run_manifest.json'), read(second, 'run_manifest.json')
assert a['status'] == b['status'] == 'passed'
for field in ['source_sha256', 'input_sha256', 'packages']:
    assert a[field] == b[field], field
assert read(first, 'metrics.json') == read(second, 'metrics.json')
pd.testing.assert_frame_equal(pd.read_csv(first / 'scored_holdout.csv'), pd.read_csv(second / 'scored_holdout.csv'), check_exact=True)
for model in read(first, 'metrics.json')['models']:
    left, right = pd.read_csv(first / (model + '_top100.csv')), pd.read_csv(second / (model + '_top100.csv'))
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert list(left.columns) == list(pd.read_csv(first / 'scored_holdout.csv', nrows=0).columns)
record = {'status': 'passed', 'verified_date': '2026-09-24',
          'input_sha256': a['input_sha256'], 'source_sha256': a['source_sha256'],
          'identical_metrics': True, 'identical_holdout_scores': True,
          'identical_alert_tables': True, 'consistent_alert_schema': True,
          'identical_package_versions': True, 'primary_seconds': a['seconds'],
          'separate_environment_seconds': b['seconds'],
          'primary_command': 'python run.py --rows 120000 --seed 42',
          'second_command': 'python run.py --rows 120000 --seed 42 --data-dir polished_run/data --out polished_run/outputs'}
Path('docs/results/reproduction.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2))
'@ | .venv-clean/Scripts/python.exe -
```

</details>

## What was checked

The behavioral tests compare all historical aggregates with an explicit pandas reference on boundary examples. They check the start of a lookback window, empty history, same-minute ties, current-amount exclusion, future-event changes, fraud-label changes, and prefix-versus-full-data equivalence. Additional tests cover chronological splitting, hand-calculated alert metrics, reproducible generation, and invalid-input rejection without replacing an existing table.

The full artifact audit checks row alignment, predictor exclusion rules, finite values, feature variation, split separation, and 6,050 historical feature values across 55 selected rows. It reloads four models and reproduces all holdout predictions, headline metrics, and top-100 transaction IDs.

These are targeted checks of the implemented event-time policy, not proof against every possible leakage mechanism or future data-feed behavior.

## Review refinements

- Fixed day boundaries and real validation-based model selection replace the inherited row-count split and unused validation set.
- Historical windows exclude the entire current minute; the documentation matches the SQL.
- Generation avoids overwriting prior planted fraud and repairs self-transfers after merchant assignment.
- One DuckDB feature worker removes the tiny floating-point aggregation differences found during initial reproduction.
- Alert exports share a consistent schema. EDA counts are serialized as integers.
- Readable function documentation, tracked evidence, generated cover artwork, and four reproducible analytical figures make the project easier to review and share.

## Scope and limitations

The figures and results come from a single simulated seed and future holdout. They do not establish JazzCash or easypaisa performance. Labels are assumed available by the training cutoff, prior events are assumed promptly observable, and network context is local. Value coverage may count the same underlying funds at multiple transaction stages. No prevented loss, recovered money, live alert policy, external-data performance, or large-scale deployment is demonstrated.
