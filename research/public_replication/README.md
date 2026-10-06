# Public replication cohort

**This replication cohort is not the cohort used to produce the sealed hackathon results. It exists so a third party can exercise the same causal data-processing and modeling pipeline using only public inputs.** Every run is labeled `PUBLIC REPLICATION COHORT — NOT SEALED HACKATHON RESULTS`. This is execution verification, not a second performance study.

The original sealed 150-case study remains **PARTIALLY REPRODUCIBLE** because its exact cohort/pilot selection and protected execution provenance are excluded. This capability needs none of those files. It never imports protected execution workers, loads original patient-level tables, restores original estimators, applies sealed calibration mappings, or updates the replay or sealed results.

## Fixed public selection

The source is [VitalDB v1.0.0 on PhysioNet](https://physionet.org/content/vitaldb/1.0.0/), DOI [10.13026/czw8-9p62](https://doi.org/10.13026/czw8-9p62), under the release's CC BY 4.0 terms. The release is anonymized and publicly available. The committed manifest includes only the selected **public dataset case numbers**, not patient/subject mappings, a private crosswalk, or the original study roster. Attribution and downloaded LICENSE remain with local inputs; project Apache-2.0 licensing does not replace the data terms.

`cohort.py` calls the **unchanged** `intraop.data.cohort_acquisition.candidate_plan` on release clinical metadata:

1. Original `population_eligibility`: adult/general anesthesia, valid overlapping surgical/recording bounds and zero recording origin.
2. Original nonempty `aline1` acquisition hint.
3. Sort integer public case IDs, apply `numpy.random.default_rng(42).permutation`, with **no pilot prefix**.
4. Take the first **12** candidates, before opening any recording/outcome.

Twelve is a small fixed size that can populate four subject groups with multiple non-training subjects while exercising six-channel processing and local model interfaces. It was chosen before outcomes were inspected, not because of AP or prevalence. No case is replaced because it lacks anchors, has no events, fails a download or fails parsing. Selection never checks membership in the private original roster; overlapping public source cases are possible and are not evidence of cohort identity.

`public_cohort_manifest.json` binds selection policy, acquisition order and canonical metadata SHA-256 values. There are no timestamps or machine paths in that manifest. Its bytes can be regenerated with `--selection-only`; metadata/schema drift fails rather than creating an alternative cohort. Models use the existing recorded configurations without tuning or hyperparameter search.

## Setup and one command

From the repository root, create a separate scientific environment using the evidence-derived existing direct pins:

```bash
cd research
python3.14 -m venv .venv-public
source .venv-public/bin/activate
python -m pip install -r requirements-data.txt -r requirements-models.txt -r requirements-dev.txt
python -m pip install --no-deps -e .
python -B scripts/run_public_replication.py --output outputs/public-replication
```

These are source-checkout commands; the adapter uses the original local `intraop` namespace through `locked_bootstrap.py`. Python 3.14.6 and the recorded direct dependency versions were used for validation. This is not a complete transitive/wheel/hardware lock; availability of compatible wheels and network access remain practical prerequisites.

The default performs public acquisition, original causal processing, subject grouping and **five conventional model executions** in fresh subprocesses. It skips both hosted TabPFN models unless explicitly requested. No private input path argument or historical/private-mode fallback exists. The destination must be a new direct child of `research/outputs/` whose name starts with `public-replication`; existing directories and symlinked destinations are rejected.

Canonical metadata and the twelve fixed `.vital` files are downloaded using the original `download_file`/`acquire_subset` functions. Four parallel recording transfers affect speed only; completion order cannot alter selection. Source SHA-256 values and the 2 GB complete selected-input cap are verified before processing. Transfer errors stop the run without replacing cases. Download size and time depend on the fixed source files and network; metadata-only selection is available before downloading recordings.

For an explicit data-only run:

```bash
python -B scripts/run_public_replication.py --output outputs/public-replication-data --skip-models
```

For independent selection verification without recording downloads:

```bash
python -B scripts/run_public_replication.py --output outputs/public-replication-selection --selection-only
```

Compare `public_cohort_manifest.json` in that output with this directory's committed manifest. `selection_completion.json` records the manifest SHA-256.

## Same scientific machinery

`runner.py` directly calls the original `build_case_tables`, `make_subject_split`, `partition_positions`, `predictor_array` and model factory. It does not implement replacement label, resampling, feature or calibration formulas.

The original endpoint remains a new MAP <65 episode for 60 consecutive represented seconds, onset in `(t,t+300]`, 60-second forecasts, `(t-300,t]` causal history, symmetric 360-second future ascertainment, original missingness/censoring and recurrent-event rearming. The exact ordered 74 predictors and 18 MAP-only predictors are checked against the original published schemas. Original tests cover causality, timestamp gaps, future perturbation, boundary labeling and censoring; adapter tests check integration and prohibit held-out/private inputs.

Splitting uses the **unchanged original** 60/15/10/15 largest-remainder subject splitter, seed 42, on subjects actually contributing eligible replication windows. There is no pilot reservation or outcome stratification. For twelve eligible unique subjects, counts are 7/2/1/2; the realized counts can differ if a fixed case has no eligible anchors or subjects repeat. Every operation from a subject remains grouped. Too few subjects or an empty partition blocks the run; no favorable-case replacement or outcome-driven split repair occurs.

The original internal partition names remain `training/tuning/calibration/test` because the original splitter uses them; output summaries prefix these with **replication_**. They are never the sealed partitions. Only replication training rows fit models. Non-training replication rows are combined in original source order as a query set for **interface verification**, with no tuning, calibration fitting or performance claim. They are not a separately powered TEST study.

The conventional outputs are prevalence, untransformed `-map_latest`, MAP-only logistic, full logistic and full XGBoost. Definitions, feature ordering, preprocessing and fixed configurations come from the existing public configuration projection and factory. Empty-feature prevalence receives dummy zero columns; it still uses all replication training labels. Learned binary models skip rather than alter the cohort if replication training contains one class. Native/process failures retain evidence and stop execution without automatic retry.

**No metrics are computed.** Outputs contain raw ranking scores/probabilities only. No sealed Platt mapping is loaded or applied. Missingness and raw probability validation use original model utilities; class 1 and query-row alignment are checked. No replication output belongs in the original README AP table.

## Optional TabPFN-3.5

Hosted execution is an explicit normal-terminal opt-in:

```bash
python -B scripts/run_public_replication.py --output outputs/public-replication-hosted --with-tabpfn
```

Initialize authentication securely in that terminal first. The adapter reads only token presence from the process environment, never sources or saves credential files, and never prints token values. Without `TABPFN_TOKEN`, TabPFN is `SKIPPED — authenticated hosted access required`; conventional/data stages still complete. An explicit opt-in prevents accidental hosted requests merely because a token is already present.

The worker requires exactly `tabpfn-client==0.6.1` and uses the original factory with `TabPFNClassifier(model_path="v3.5_default", random_state=42)`. MAP/full contexts use all replication training rows with exactly 18/74 predictors, respectively, no manual preprocessing or alternate model. Classes must be `[0,1]`; positive probability is column 1. Provider alias/defaults and hosted access remain external dependencies. Do not infer a hosted success from a mocked test. No hosted call occurred during validation of this release.

`--model-python` permits an explicitly named worker interpreter when data and model dependencies use separate environments. There is no silent interpreter fallback. Normal users installing both requirement sets into `.venv-public` can omit it.

## Outputs, reruns and privacy

```text
outputs/public-replication/
  scope.json
  public_cohort_manifest.json
  source/vitaldb-1.0.0/                 # downloaded public inputs, local only
  tables/public_replication_*.csv       # labels/features/groups/audits, local only
  tables/public_replication_split.json  # replication subject mapping, local only
  public_model_inputs.npz               # predictors/labels only, local only
  models/<model>/public_raw_predictions.npz
  models/<model>/completion.json
  models/<model>/exit.json
  public_replication_summary.json       # aggregate counts/statuses/table hashes
```

`failure.json` contains only stage/exception class when orchestration stops. Worker logs retain sanitized status, never arbitrary exception messages, patient values or hosted response metadata. There is no automatic retry, resume or overwrite. A new explicit output path is required for another public validation run. All generated inputs/tables/predictions/estimators/caches remain ignored; only the public selection manifest, adapter code, tests, documentation and aggregate release-validation summary are committed.

To repeat deterministic processing without another download, after the first run:

```bash
python -B scripts/run_public_replication.py \
  --output outputs/public-replication-repeat \
  --reuse-public-source outputs/public-replication/source/vitaldb-1.0.0
```

Only an explicit prior **public** source directory under the owned output root is accepted, with its scope marker, matching public selection and canonical release hashes. The I/O boundary allows that source and identity marker; prior outcomes/model outputs, private rosters, crosswalks, protected approvals, historical calibration and sealed bundles remain blocked. No hidden local cache is searched. Inputs are copied into the new run, not mutated. Repeated table fingerprints can then be compared; hosted outputs are not promised bitwise identical.

## Validation

Run public synthetic/unit checks and formatting from `research/`:

```bash
python -B scripts/run_public_tests.py
python -m ruff check . --per-file-ignores historical_tests/test_modeling_process_isolation.py:I001
python -m ruff format --check .
python -B scripts/run_public_replication.py --help
```

The existing archived-test lint exception preserves its original bytes. Original historical artifact-dependent tests are unchanged and still require their private historical inputs. Public selection determinism, fail-closed paths, synthetic full-pipeline repetition, group disjointness, original source hashes and mocked exact TabPFN calls are covered separately. `validation_summary.json` records the actual public-data execution and repeatability checks for this release, without patient-level values or headline metrics.

A runnable replication capability does not change the sealed study's honest **PARTIALLY REPRODUCIBLE** classification, nor its counts or reported AP values.
