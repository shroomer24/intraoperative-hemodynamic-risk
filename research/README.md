# Scientific pipeline and reproducibility audit

This subtree recovers the actual implementation behind the sealed study. It is **partially reproducible from the public repository**, not a new experiment and not a claim that a fresh checkout can recreate every sealed artifact.

The reader, preprocessing, endpoint, anchor construction, features, dataset and subject splitter match the cohort-generation source hashes. Original development/locked workers, model definitions and calibration/evaluation functions are also retained. `provenance/source_manifest.json` records original and published SHA-256 values, byte identity, and available generation/lock bindings. Original artifacts under `reference/` are distinguished from explicitly identified public-safe projections.

Two existing files have documented publication adaptations: the acquisition CLI requires the private pilot cases explicitly instead of embedding their identifiers; the replay exporter takes explicit source/destination paths instead of a private machine path. New `reproduce_tables.py`, public-test runner, synthetic-interface smoke and export shell wrapper are post-publication adapters. They are not claimed to have produced the sealed experiment. No original scientific formula was reconstructed or repaired.

## What is missing for complete public reproduction

The exact 150-case source roster and ten inspected pilot subjects are private runtime inputs. The public repository does not contain their identifiers, a crosswalk, clinical metadata, raw recordings, complete feature/label tables, fitted estimators or patient-level calibration checkpoints. The original protected execution plans, approvals, receipts and complete hash-bound source bundles also remain local; do not replace them with the public-safe projections.

Consequently, deterministic processing and splitting are reproducible **conditional on the exact private selection**, but exact cohort membership is not recoverable from this publication alone. Repeating a time-limited acquisition or choosing ten different pilot cases creates a different study. Renaming subjects before the sorted-ID splitter can also change assignments. We have not supplied a fabricated replacement roster.

TabPFN requires authenticated Prior Labs hosted access. Its alias and server defaults are external dependencies; future responses are not guaranteed bitwise identical. The concrete development checkpoint identity was not retained, so continuity between development and the subsequently verified concrete calibration/TEST checkpoint is not proven. There is no local model weight snapshot in this repository.

A fresh approved execution plan for a separate replication run is not supplied here. The historical locked workers deliberately require their original private approval/hash chain. Their one-time guards have not been weakened to turn a completed study into a rerunnable job. Source-level recovery does not imply that the historical commands can run from a public checkout without those inputs.

## Scientific task and source

The endpoint is a **new** invasive-MAP episode beginning in `(t,t+300]`, with MAP **strictly below 65 mmHg for 60 consecutive represented seconds**. The history is `(t-300,t]`; forecast times are multiples of 60 seconds relative to recording start.

Source: [PhysioNet VitalDB v1.0.0](https://physionet.org/content/vitaldb/1.0.0/), DOI [10.13026/czw8-9p62](https://doi.org/10.13026/czw8-9p62). Acquisition uses that release's file URLs, checksum list and data license, not a replacement API dataset. Data remain governed by the source terms, independently of this project's Apache-2.0 source license.

The original downloader in `src/data/acquisition.py` retrieves `LICENSE.txt`, `SHA256SUMS.txt`, `clinical_data.csv`, `track_names.csv`, `clinical_parameters.csv` and explicitly requested `.vital` recordings. It verifies release hashes, limits transfers and accepts at most 20 case IDs per subset call. The full raw dataset is not committed.

Expected local layout, created only by an operator acquisition/restoration:

```text
data/raw/vitaldb-1.0.0/
  LICENSE.txt
  SHA256SUMS.txt
  clinical_data.csv
  track_names.csv
  clinical_parameters.csv
  vital_files/*.vital
private/cohort-selection.json  # not distributed; never commit
outputs/                      # new replication results only
```

These are real runtime paths accepted by the commands below, not files claimed to be included. The required selection file has exactly `completed_case_ids` and `development_subjects_pinned_to_training`; it is the original selection, not invented example identifiers.

`reference/track_units.json` verifies the actual numeric track dictionary:

| Internal channel | VitalDB numeric track | Unit |
| --- | --- | --- |
| map | Solar8000/ART_MBP | mmHg |
| sbp | Solar8000/ART_SBP | mmHg |
| dbp | Solar8000/ART_DBP | mmHg |
| hr | Solar8000/HR | /min |
| spo2 | Solar8000/PLETH_SPO2 | % |
| etco2 | Solar8000/ETCO2 | mmHg |

The pinned official reader is `vitaldb==1.7.2`. `src/data/vitaldb_reader.py` uses original numeric `track.recs`, avoiding convenience resampling APIs that floor observation times or interpolate. Header/all-record timestamp inspection establishes the anonymized recording origin; it is coordinate conversion, not future physiological normalization or recovery of actual calendar dates. Relative seconds, not fabricated calendar times, are used throughout. Track-name metadata are a dictionary, not per-case availability evidence.

## Cohort, chronology and censoring

Original population eligibility requires adult age (including the release's `>89` top coding), `ane_type == "General"`, finite ordered surgical bounds overlapping recording availability, and `casestart == 0`. An existing arterial-line hint orders acquisition; actual numeric ART_MBP availability is verified after reading. Optional signals do not independently exclude a case. There is no invented operation-name classifier or new clinical normal-range filter.

Acquisition is outcome-blind: a fixed pilot prefix plus sorted arterial-line-hinted candidates shuffled with NumPy seed 42. It stops at the target/cap/time/first failure, without replacing cases based on outcomes. The final 150-case completed prefix, not a new clock-limited download, defines v0.3. The historical v0.1/v0.2 acquisition ledgers are not redistributed.

`preprocessing.py` represents integer surgical seconds using the latest valid observation at or before that second, held at most ten seconds (inclusive). No future interpolation/backfill, centered smoothing or whole-operation physiological normalization is used. Duplicate timestamps retain the last valid original-order observation. MAP must be finite, positive and at most 250 mmHg; optional base tracks receive nonfinite masking. Measurement age remains recorded after values become stale. SBP/DBP/MAP ordering inconsistencies are audited rather than silently removed. Represented seconds stop at recording/surgical availability.

`labels.py` confirms onset `s` at `s+59` for low seconds `s,...,s+59`. Missingness breaks the low run. After confirmation, a recurrent onset requires **60 consecutive valid seconds with MAP >=65** to rearm; brief recovery or missingness does not rearm. Episode state resets per operation.

An eligible forecast needs all 300 history grid seconds, at least 270 valid MAP seconds, and every recent second `t-59,...,t` valid and MAP >=65. Both positive and negative labels require **all 360 future MAP seconds** `t+1,...,t+360` and sufficient remaining surgery. Missing future MAP or insufficient observation censors a causally eligible anchor; censored rows never become label-zero examples. `y=1` means a confirmed onset with `t < s <= t+300`. An onset at `t` is excluded; one at `t+300` is included. An episode is evaluable if it has at least one eligible uncensored anchor in `[s-300,s)`.

Future information participates only in retrospective target ascertainment/censoring, not in X. Complete future-observation selection can still limit generalizability. These are observed low-MAP outcomes under historical treatment, not untreated deterioration or guaranteed five full minutes of advance warning. Human waveform adjudication remained pending.

## Features and patient-disjoint splits

The exact ordered schema is `reference/feature_schema.json`; model subsets are `reference/feature_sets.json`. `src/features/core.py` produces 74 features:

- Eight summaries for each of six channels: latest finite held value, 60/300-second medians, population SD, 10th/90th percentiles, and 60/300-second raw-observation OLS slopes.
- MAP minute medians (oldest minute 1 through latest minute 5), latest-minus-previous-minute median, and valid-time fractions below 70/75.
- Pulse pressure summaries/slope and HR/SBP ratios.
- Missing fractions and measurement ages for all channels.
- Causally known elapsed recording seconds.

Raw slopes need at least two distinct observation times; missing observations do not compress time. Ratios require positive SBP. Native missingness remains NaN until an explicitly specified model preprocessing step. `docs/feature_schema.md` describes the original formulas. MAP-only is the **18 ordered `map_` columns**, not a newly selected subset. Elapsed seconds are a prespecified known relative-time predictor; they are not a future timestamp or patient identifier.

`pipeline.py` filters every raw feature window to `(t-300,t]`, enforces contiguous integer history and separates predictors from label/metadata tables. No source patient/case IDs, future onset fields or surgical-end information enter X. Group IDs exist locally only for splitting/auditing.

`src/evaluation/subject_splits.py` uses sorted **string** subject IDs, NumPy seed 42, 60/15/10/15% quotas and largest-remainder rounding, with partition-order tie breaking. All operations from a subject share one assignment. Ten inspected pilot subjects are pinned to training, triggering the explicit quota-preserving unstratified fallback. It is not a calendar holdout and not an outcome-stratified final split. The frozen population yields 90/23/15/22 subjects; manifest overwrite with different assignments is refused.

## Models and TabPFN-3.5 provenance

The exact factory is [`src/models/benchmarks.py`](src/models/benchmarks.py). Original process-separated development is in `scripts/modeling_development_resume.py`, `modeling_xgboost_worker.py`, `modeling_tabpfn_worker.py` and their support modules. Original initial runner and its scientific selection/export functions are retained for provenance. The initial mixed-process run crashed at XGBoost; approved process isolation completed the original candidates without changing scientific configuration.

| Model | Inputs / preprocessing | Selected configuration |
| --- | --- | --- |
| prevalence | All TRAINING labels, constant probability | 498 / 13,975 |
| current_map | map_latest, score only | `-map_latest`, no probability transform |
| logistic_map | 18 MAP features; TRAINING median imputation, empty columns retained, StandardScaler | L2, C=10, lbfgs |
| logistic_full | 74 features; same TRAINING-only preprocessing | L2, C=1, lbfgs |
| xgboost | 74 features, native missingness | 200 trees, depth 2, learning rate .05, hist, n_jobs=1 |
| tabpfn_map | 18 MAP features, no manual preprocessing | v3.5_default, random_state=42 |
| tabpfn_full | 74 features, no manual preprocessing | v3.5_default, random_state=42 |

Every remaining hyperparameter is recovered in `reference/selected_models.json`; the prespecified candidate grid is `reference/development_matrix.json`. Logistic C candidates are .01/.1/1/10; XGBoost has eight fixed combinations of 200/400 trees, depth 2/3, rate .05/.1. Selection uses TUNING AP, then prespecified lower tuning log loss, simplicity rank and lexical ID ties. Neither CALIBRATION nor TEST is available to the development loader. The final frozen TRAINING context is all 13,975 rows, 90 subjects and 498 positives; no enrichment, resampling or new feature selection was introduced.

The exact hosted construction is:

```python
from tabpfn_client import TabPFNClassifier
model = TabPFNClassifier(model_path="v3.5_default", random_state=42)
model.fit(training_predictors, training_labels)
probabilities = model.predict_proba(query_predictors)
```

That actual factory call, plus feature-whitelisting, fit/predict and identity validation, is visible in the original workers. XGBoost and hosted client/Torch runtimes are kept in separate fresh processes. No chunked 5,000-row approximation or alternate TabPFN generation replaces the 13,975-row context. Query rows preserve frozen source order; classes must be `[0,1]`, binary matrices must match row counts, and positive-class probability is column 1. Native client/server behavior handles missingness. Unset inference parameters use provider defaults; those defaults are not a complete pinned model configuration.

Use `tabpfn-client==0.6.1`, not the stale 0.3.3 environment artifact from the pre-access block. Authenticated hosted execution requires a token supplied securely outside Git, from an operator's normal terminal. Do not echo/log/save tokens or publish authentication files. No local checkpoint is required or included. No hosted call was made during this audit.

`reference/hosted_identity.json` records the safe retained TEST identity: requested alias `v3.5_default`, returned `/app/tabpfn_models/tabpfn-v3.5-20260909.safetensors`, billing `v3.5`, execution `standard`. The unchanged compatibility code accepts only the approved alias/concrete path plus required class/data/configuration checks. The originally stricter validator and reviewed execution amendment are separate original files; neither was silently relaxed. The abandoned bounded representation study is retained only as historical code, not a recommended rerun or an extra model selection step.

## Calibration, evaluation and sealed evidence

`src/evaluation/calibration.py` implements the prespecified independent Platt mappings for logistic_map, logistic_full, xgboost, tabpfn_map and tabpfn_full. CALIBRATION has 2,393 windows and 128 positives from 15 subjects. Inputs are clipped to `[1e-6,1-1e-6]`, logit-transformed, then fit by LogisticRegression(C=1, solver=lbfgs, intercept=True, max_iter=2000, tol=1e-4, class_weight=None, random_state=42). No calibration-method/threshold/model selection occurs there. Prevalence is not recalibrated; current MAP remains an untransformed score. The five original numeric mappings are retained byte-identically in `reference/calibration_parameters.json`.

The actual aggregation worker is `artifacts/modeling-v01/calibration-aggregation-v01/aggregation_worker.py`. Recovery-specific MAP/full workers preserve raw immutable probability checkpoints before optional metadata/publication; final TEST uses the sealed mappings without refitting Platt. Conventional estimators are reused from their TRAINING fits saved during calibration; hosted models re-establish the same TRAINING context. TEST is reserved before values are parsed, verifies the lock/protocol/source/approval chain, and cannot automatically retry once execution evidence exists.

The final worker is `artifacts/modeling-v01/final-test-v01/final_test_worker.py`; reporting is in `final_test_reporting.py`. AP uses `sklearn.metrics.average_precision_score` on binary positive-label-1 scores. Raw AP is primary; calibrated probability metrics are secondary. Eligible uncensored windows are individual prediction instances. Subject bootstrap resamples all windows of each sampled subject, preserving multiplicity, for 1,000 seed-42 replicates; single-class replicates are undefined for AP/AUROC and counted explicitly. These are descriptive uncertainty summaries, not proof of clinical superiority or significance.

`reference/sealed_test_metrics.json` is copied byte-for-byte from the original sealed metrics. No TEST prediction or metric was recomputed. The documented raw AP values round to XGBoost .1858, MAP-only TabPFN .1839 and full TabPFN .1740. They agree with the README. The source lock itself is not replaced by `selected_models.json`: that file is explicitly a public-safe configuration projection, not an executable production lock.

### Count provenance

| Quantity | Value | Verification |
| --- | ---: | --- |
| Subjects / operations | 150 / 150 | Independently tallied frozen metadata |
| Eligible windows / positive windows | 22,707 / 836 | Independently tallied frozen labels/metadata |
| Confirmed / evaluable episodes | 296 / 247 | Independently tallied frozen episode table |
| Monitored MAP hours | 477.65166666666664 | Frozen report verified; not rebuilt for all 150 cases |
| Training / tuning / calibration / test subjects | 90 / 23 / 15 / 22 | Independent grouping tally and manifest membership audit |
| TEST windows / positives | 3,146 / 116 | Independent frozen-table tally plus sealed receipt |
| Raw AP values above | .1858 / .1839 / .1740 | Original sealed metrics verified, not re-evaluated |

`reference/independent_count_audit.json`, `verified_counts.json` and `training_case_reconstruction_audit.json` state the exact verification scope. Counting frozen tables is not independent regeneration of the full study. The count audit parsed frozen grouping/label/episode fields; no held-out raw physiology or model inference was used. All subject intersections are empty. One TRAINING case independently reconstructed 82 anchors: labels, anchor times, event onsets and episode evaluability match exactly; 74 feature columns match with NaNs equal and rtol=1e-12 / atol=1e-10 for CSV floating-point round trips. No patient identifier or patient-level feature value is retained in the public audit summaries.

## Replay export and immutable frontend boundary

`export/export_replay.py` is the actual Phase 0 exporter with only path initialization adapted; its full original/published hashes and exact changes are documented. It verifies the original lock, final receipt, query order and protected source chain, blocks model imports/network/scientific writes, copies frozen predictions and displays existing causal observations. It does not redetect outcomes, refit models, refit calibration or regenerate scientific labels. The private source/display crosswalk stays outside the public tree.

The frontend already contains the 68 approved deidentified files under `../public/replay-v01/`, across 22 presentation-ordinal cases. The complete original local export has 90 files; 22 `anchor-status.json` artifacts were never publicly shipped and remain private/local. That difference is explicit, not a missing public scientific evaluation. No original public replay data or runtime code changed in this publication.

A full original sealed source bundle is required to rerun the exporter; the public-safe projections are insufficient for its hash guards. It refuses to overwrite an existing output, including empty directories. Surgical-context presentation metadata are maintained separately from the scientific predictors.

## Environment and real commands

Python 3.14.6 was used for both cohort and modeling in two separate original environments. The data runtime had VitalDB 1.7.2 and Matplotlib 3.11.2; the recorded final model lock/test runtime pins NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.0, XGBoost 3.3.0, SciPy 1.18.0, joblib 1.5.3 and tabpfn-client 0.6.1. The original pyproject retains its broader supported range; that range is not evidence that all versions were tested. Requirements files are evidence-derived direct pins, not a complete transitive/wheel/hardware lock. Original native failures on macOS demonstrate why process isolation and platform validation matter.

Frontend dependencies remain solely in the root package manifest/lock. The research package has its original namespaced `intraop` layout and original pyproject. From the repository root, use a **separate** research environment:

```bash
cd research
python3.14 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements-data.txt -r requirements-dev.txt
python -m pip install --no-deps -e .
python -B scripts/run_public_tests.py
python -m ruff check . --per-file-ignores historical_tests/test_modeling_process_isolation.py:I001
python -m ruff format --check .
python scripts/reproduce_tables.py --help
```

The single archived-test import-order lint exception preserves its original bytes after archival relocation; it is not a production-code exemption. `historical_tests/` preserves original integration tests requiring excluded private plans/registries/approvals. It is deliberately outside the public synthetic suite and is **not** claimed passing from this public checkout. An initial audit run exposed those missing dependencies; they were not replaced with fake ledgers to manufacture passing tests.

**Only with the exact original private selection supplied** and permission to acquire/rebuild local data, the real path adapter is:

```bash
python -B scripts/reproduce_tables.py \
  --source data/raw/vitaldb-1.0.0 \
  --selection private/cohort-selection.json \
  --output outputs/reproduced-cohort \
  --acquire
```

Omit `--acquire` if a complete checksum-verified local source already exists. It calls the original downloader in <=20-case batches and the original `run_tables`, seed 42, with the original pinned pilot subjects. The destination must not exist. It checks regenerated table hashes against `reference/table_manifest.json`; mismatches remain evidence, never a replacement for sealed outputs. This full command was **not executed** during the audit. It cannot reproduce exact membership without its withheld input. The one-case validation above demonstrates a narrower verified reconstruction, not a claim that the full command has been validated on all cases.

For synthetic conventional interfaces only, in a separate model environment:

```bash
python3.14 -m venv .venv-models
source .venv-models/bin/activate
python -m pip install -r requirements-models.txt
python -m pip install --no-deps -e .
python -B scripts/interface_smoke.py --model prevalence
python -B scripts/interface_smoke.py --model logistic_map
python -B scripts/interface_smoke.py --model logistic_full
python -B scripts/interface_smoke.py --model xgboost
```

Each command starts a fresh process, uses deterministic synthetic data and recorded selected hyperparameters, and cannot invoke TabPFN or load scientific tables. The public tests mock the exact TabPFN constructor and validate the class-1 probability contract offline. They do not establish current hosted access.

Historical runner CLI inspection is available without fitting:

```bash
python scripts/modeling_development.py --help
python scripts/run_locked_evaluation.py --help
```

The proper historical order was acquisition -> tables/split -> TRAINING/TUNING development and selection -> lock -> approved raw CALIBRATION -> five Platt mappings -> one-time approved TEST -> immutable replay export. Restoring private plans/approvals and creating a separately reviewed replication capability are prerequisites for a new empirical rerun. Do **not** rerun the historical completed directories or fabricate approval JSON from these public artifacts.

For export only when a full original sealed bundle has been provided at `private/original-scientific-bundle/`, under the data environment:

```bash
bash scripts/export_original_bundle.sh private/original-scientific-bundle outputs/replay-export-app
```

That is a real wrapper, not an instruction to recreate hidden files. The named private bundle is not included and the command fails closed without it. Outputs and the adjacent private audit/crosswalk remain local and must never be copied into the public frontend. No export execution was performed during this audit.

The frontend checks remain, from the repository root:

```bash
npm ci
npm test
npm run typecheck
npm run build
```

## Stage provenance and final determination

| Stage | Recovery classification | Public reproduction limit |
| --- | --- | --- |
| VitalDB ingestion | ORIGINAL IMPLEMENTATION FOUND | Public source and pinned reader; raw data downloaded separately |
| Cohort construction | ORIGINAL IMPLEMENTATION FOUND | Exact roster/pilot input withheld; membership partial |
| Event detection | ORIGINAL IMPLEMENTATION FOUND | Requires source recordings; original code and synthetic tests |
| Anchor generation | ORIGINAL IMPLEMENTATION FOUND | Same frozen cadence/history rules |
| Labels | ORIGINAL IMPLEMENTATION FOUND | Exact endpoint; one TRAINING case compared |
| Censoring | ORIGINAL IMPLEMENTATION FOUND | Original symmetric 360-second rule |
| Full feature generation | ORIGINAL IMPLEMENTATION FOUND | Exact ordered 74-schema and matching core source hashes |
| MAP-only features | ORIGINAL IMPLEMENTATION FOUND | Exact 18-column ordered subset |
| Patient splits | ORIGINAL IMPLEMENTATION FOUND | Exact subjects/pilot reservation required; disjointness audited |
| Logistic baselines | ORIGINAL IMPLEMENTATION FOUND | Configs/factory/isolated execution recovered; private run plans absent |
| XGBoost | ORIGINAL IMPLEMENTATION FOUND | Exact selected config and isolation recovered; native/runtime dependency |
| TabPFN-3.5 | ORIGINAL IMPLEMENTATION FOUND | Authenticated external service, alias/defaults and historical identity gap |
| Calibration | ORIGINAL IMPLEMENTATION FOUND | Code, protocol and original numeric mappings; private raw checkpoints absent |
| TEST evaluation | ORIGINAL IMPLEMENTATION FOUND | Source/protocol and ORIGINAL ARTIFACT FOUND metrics; private one-time chain absent |
| Replay export | ORIGINAL IMPLEMENTATION FOUND | Explicit path adapter; complete original sealed bundle required |

All scientific stages were found; none is labeled an original stage solely on the basis of a new plausible implementation. The **public run chain** is PARTIALLY RECOVERABLE because inputs/execution provenance are intentionally excluded and exact hosted inference is not guaranteed. No scientific stage reconstruction was needed. Existing README wording remains untouched; the appended root section clarifies that recovered research source is now included while the presentation and scientific execution remain separate.

The README's listed counts/AP values agree with the checked artifacts. Important provenance differences are: raw vs calibrated metric status, the 360-second ascertainment requirement despite a 300-second onset horizon, mandatory 60-second rearm, private acquisition/pilot identity, unstratified pinned-subject splitting, and unavailable concrete development model identity. These details are now explicit; the experiment was not changed to simplify them.

**PARTIALLY REPRODUCIBLE**
