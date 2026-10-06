# Intraoperative Hemodynamic Risk

**A historical replay system for predicting new sustained intraoperative hypotension within the next five minutes.**

**Live demo:** https://intraop-hemodynamic-risk.pages.dev/demo

This started with a  simple question: **Can a tabular foundation model make useful near-term predictions from continuous operating-room data?**

I really wanted to explore this because my brother is a MS4 at McGovern Medical School and he wants to get into General Surgery. We had always chatted over phone calls about the gnarly rotations he used to go through where shifts used to easily reach 14 hours long back to back. What amazed me was his ability to be able to intake so much information at once, and be able to act on a whim based on all the calculations he had ran in his head. This left me reflecting on if I could make something that streams patient data into a layer where the state of the patient could be analyzed on in real time and provide valuable insight as a helper tool for doctors.

The raw data aren't tabular. They are continuous physiologic signals collected during surgery. I built a causal feature pipeline that turns the previous five minutes of those signals into one structured patient-state row, then compared TabPFN-3.5 against simpler baselines and XGBoost.

The main thing I cared about was keeping the experiment honest. Every prediction only uses information that would have been available at that point in the surgery. The held-out patients stay completely separate from training, and the frontend replays frozen test predictions instead of quietly recomputing them.

## The problem

Intraoperative hypotension can develop quickly during surgery. Looking at one blood-pressure value tells you what is happening now, but it does not necessarily tell you what is about to happen.

For this project, I defined one narrow prediction task:

> **Will a new sustained hypotensive episode begin within the next five minutes?**

A sustained episode is defined as invasive **mean arterial pressure (MAP) below 65 mmHg for at least 60 continuous seconds**.

The model makes a new prediction every 60 seconds using only the previous five minutes of observed physiology.

This is intentionally a specific endpoint. I am not trying to predict every kind of patient deterioration, shock, mortality, or treatment need.

## What I built

The pipeline looks like this:

```text
VitalDB continuous monitor data
        ↓
causal five-minute history
        ↓
74 engineered patient-state features
        ↓
TabPFN-3.5 + conventional baselines
        ↓
P(new sustained hypotension within 5 min)
        ↓
held-out historical replay
```

The six physiologic channels used in the full feature set are:

- MAP
- systolic blood pressure
- diastolic blood pressure
- heart rate
- SpO₂
- end-tidal CO₂

The frontend mainly shows MAP, heart rate, SpO₂, and ETCO₂ because those are the easiest signals to follow during a replay. The model itself uses **74 engineered predictors across all six channels**.

Those predictors describe things like current values, one and five minute summaries, recent slopes, variability, measurement age, and recent MAP behavior.

## Why the causal setup matters

The difficult part of this project was not just getting a model to output a probability. It was making sure that probability was based only on information that would have existed at that exact time.

For a prediction made at time `t`:

- features only use data at or before `t`
- no future interpolation or backward filling is allowed
- future outcome information is not included in the feature table
- whole-operation normalization is not used
- patient IDs and case IDs are not model features

The frontend follows the same idea. In **Live Replay**, future physiology and future outcomes stay hidden. Once enough time has actually passed to confirm a 60-second hypotensive episode, the interface can reveal it. **Review** mode is the explicit hindsight view.

## Data

I used **VitalDB v1.0.0**, an open perioperative dataset containing high quality physiologic data from surgical cases.

Final study cohort:

| Split | Patients / operations |
| --- | ---: |
| Training | 90 |
| Tuning | 23 |
| Calibration | 15 |
| Test | 22 |
| **Total** | **150** |

Across the full cohort, the frozen dataset contains:

- **22,707 eligible prediction windows**
- **836 positive windows**
- **296 confirmed hypotensive episodes**
- **247 evaluable episodes**
- about **477.7 monitored hours**

The held out test set contains **3,146 prediction windows**, including **116 positive windows**.

The split is patient-disjoint. This basically means a patient can't contribute windows to both training and test.

## Models

I didn't want to compare TabPFN against weak baselines, so the final comparison includes:

1. training prevalence
2. current MAP
3. MAP-only logistic regression
4. full-feature logistic regression
5. full-feature XGBoost
6. MAP-only TabPFN-3.5
7. full-feature TabPFN-3.5

The MAP-only comparison is useful, because it asks whether the model is doing more than reacting to recent blood pressure. The full feature models then test whether the broader physiologic state adds useful information.

## Evaluation

The primary metric is **Average Precision (AP)**.

That matters because the held-out test set is imbalanced: only 116 of 3,146 eligible windows are positive. A model could get high raw accuracy by predicting "no event" most of the time, so accuracy is not the main metric to worry over.

Sealed held-out AP:

| Model | Test AP |
| --- | ---: |
| XGBoost, full physiology | **0.1858** |
| TabPFN-3.5, MAP only | **0.1839** |
| TabPFN-3.5, full physiology | **0.1740** |

XGBoost had the best AP on this held-out cohort. I kept that result as-is rather than tuning around it and trying to milk the other models. As stated in the rules and following what any sensible person would do, my goal wasn't to force a TabPFN win; it was to test whether a tabular foundation model could work on a causal representation of continuous perioperative data and compare it fairly against strong conventional baselines.

## Demo

The public app is a **historical replay**, not a live clinical system.

**Demo:** https://intraop-hemodynamic-risk.pages.dev/demo

The curated demo opens on **Case 019**, a videoscopic thoracic metastasectomy. At the selected retained forecast anchor, the full physiology TabPFN model shows a **30.4% calibrated probability** of a new sustained hypotensive episode beginning within the next five minutes.

From there you can:

- play or pause the surgery
- scrub through surgical time
- switch among all 22 held-out cases
- switch among all seven model comparisons
- view retained raw or calibrated probabilities where available
- compare models at the exact same forecast anchor
- switch between causal **Live Replay** and hindsight **Review**
- inspect the surgical context for each held-out case

The app uses frozen held-out predictions. It does not call TabPFN or another model while you are using the site.

## Frontend

The replay interface is built with:

- React
- TypeScript
- Vite
- uPlot
- Argent / Paper presentation components
- Cloudflare Pages

The production site is static. There is no backend inference service attached to the demo.

### Run it locally

Node 22+ is recommended.

```bash
npm ci
npm run dev
```

Then open the local Vite URL shown in the terminal.

Useful checks:

```bash
npm test
npm run typecheck
npm run build
```

## Repo layout

```text
src/
├── components/        # selectors, replay controls, prediction UI
├── charts/            # physiology and risk-history plots
├── data/              # frozen replay loading and validation
├── replay/            # replay clock and anchor logic
├── outcomes/          # causal event/outcome visibility
├── case-context/      # safe surgical context shown in the UI
├── demo/              # curated /demo starting state
├── metal/             # presentation effects
└── styles/

public/replay-v01/     # frozen presentation data
tests/                 # frontend and replay behavior tests
docs/                  # demo/release notes
release/               # static release assets and headers
```

The scientific training/evaluation pipeline was kept separate from the presentation so any UI work could not silently retrain models or mess with the sealed test results.

## Reproducibility and guardrails

A few choices were intentional:

- test predictions were frozen before the replay UI was built
- patient groups are disjoint across train/tune/calibration/test
- the test set was not used to choose model configurations
- calibration used a separate calibration partition
- current MAP stays a ranking score, not a fake probability
- missing future observation is censored instead of being called a negative
- public case labels are presentation ordinals rather than source patient identifiers
- the public build does not contain the private case crosswalk or source patient IDs

The final public release was also checked across desktop, tablet, and mobile layouts with no new browser console errors.

## Limitations

This is an **exploratory retrospective research prototype**, not a clinically validated medical device tool.

Important limitations:

- the cohort is relatively small
- the data come from one public perioperative source
- repeated windows within one surgery are correlated
- historical clinician actions affect the observed physiology and outcomes
- there is no prospective validation
- there is no external-hospital validation
- the five-minute horizon means an event can begin anywhere inside that window; it does not guarantee five full minutes of warning

So the result should be read as a proof of a research framework, not a claim that this system is ready for clinical use.

## Data and attribution

VitalDB v1.0.0 is available through PhysioNet:

https://physionet.org/content/vitaldb/1.0.0/

VitalDB paper:

> Lee, H.-C. et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* Scientific Data 9, 279 (2022).

Third-party software notices used by the frontend are listed in `THIRD_PARTY_NOTICES.md`.

## License

The project's own source code is released under the [Apache License, Version 2.0](LICENSE).

Third-party software, frontend dependencies, VitalDB/PhysioNet data, and TabPFN/Prior Labs model assets remain subject to their respective licenses and terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained software notices.

## Reproducing the experiment

Recovered scientific code, frozen schemas/configurations, and provenance are available under [`research/`](research/). See [`research/README.md`](research/README.md) for validated steps, environment setup, and reproduction commands.

Exact reproduction of the sealed experiment remains partial: private cohort/execution inputs are excluded, and identical hosted TabPFN responses are not guaranteed. The scientific and presentation pipelines remain separate; the recovered research source is now included in this repository.

## Interactive demo walkthrough

If you're reviewing this project for the hackathon, I made a short step-by-step walkthrough for the live demo:

**[Follow the interactive demo walkthrough →](docs/DEMO_SCRIPT.md)**

It walks through the held-out Case 019 replay, what the TabPFN prediction means, Live Replay vs Review, the model comparisons, and the main results.

**Live demo:** https://intraop-hemodynamic-risk.pages.dev/demo
