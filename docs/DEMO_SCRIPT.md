# Interactive Demo Walkthrough

Open the live demo and follow along:

**https://intraop-hemodynamic-risk.pages.dev/demo**

This should take about 3–5 minutes.

## 1. What am I predicting?

> Based only on physiology available right now, will this patient enter a new sustained hypotensive episode within the next five minutes?

Here, sustained hypotension means **invasive MAP <65 mmHg for at least 60 continuous seconds**. One blood-pressure value tells you what is happening now, but not necessarily what is about to happen.

My research question: **Can a tabular foundation model make useful near-term predictions from continuous operating-room data?**

## 2. Start with Case 019

Leave the starting state unchanged: **Held-out operation → Case 019**, **Model → TabPFN-3.5 · Full physiology**, and **Live Replay**.

The surgical context reads **Metastasectomy · Thoracic surgery · Videoscopic**. The replay starts paused at **00:12:54**, with a **30.4% calibrated probability** in the **Prediction** panel.

That is the model's estimated probability of a *new sustained episode beginning* after that forecast and within the next five minutes. It is not the probability of low MAP right now, or a guarantee of an event. Calibration used a separate patient partition.

## 3. What does the model actually see?

Look at **Physiology**: MAP, heart rate, SpO₂, and ETCO₂. The full model uses six channels: those four plus systolic and diastolic blood pressure.

The preceding five minutes become **74 causal patient-state predictors**: current values, one- and five-minute summaries, slopes/trends, variability, recent MAP behavior, missingness, and measurement age. Surgical context helps you follow the case; it is not a model feature.

## 4. Step 1 — Press Play

Press **Play** and watch MAP. The default **30×** playback compresses surgical time; press **Pause** whenever you want to inspect it.

Although this is historical, **Live Replay behaves causally**: future physiology and future outcomes remain hidden.

The first drop below 65 does not immediately count as sustained hypotension. It must last **60 continuous represented seconds**. Only after that duration has elapsed can the interface confirm the event and retrospectively identify its onset. Watch **Historical events** and the **Historical outcome** inside the Prediction panel.

The **Time since surgical start** slider lets you scrub. Seeking pauses playback; the displayed forecast remains tied to its retained issue time, rather than refreshing continuously.

## 5. Why five minutes of prediction but six minutes of observation?

The prediction horizon is five minutes. Retrospectively, however, the pipeline requires six minutes of future MAP observation to determine the label. An episode can begin near the very end of the five-minute forecast window, so another 60 seconds are needed to verify that it truly remained below 65 long enough to count as sustained hypotension.

**Those future observations are used only to determine the historical outcome. They never enter the model predictors.**

## 6. Step 2 — Switch from Live Replay to Review

Pause, then select **Review**.

- **Live Replay** = what could have been known at that moment.
- **Review** = explicit historical hindsight.

Review exposes future physiology within the chart's visible time range and the selected forecast's actual outcome. You can scrub to explore the trajectory and open **Event details** for onset and confirmation times.

**None of the future information shown in Review was available to the model when the forecast was issued.**

## 7. Step 3 — Open Compare models

Open **Compare models** below the playback controls. All estimates align to the **same retained forecast anchor**, not different moments in surgery.

The seven comparisons are:

1. TRAINING prevalence — a fixed baseline.
2. Current MAP — an untransformed **−latest MAP ranking score**, not a probability.
3. Logistic · MAP — MAP-only logistic regression.
4. Logistic · Full physiology — full-feature logistic regression.
5. XGBoost · Full physiology — full-feature XGBoost.
6. TabPFN-3.5 · MAP — MAP-only TabPFN.
7. TabPFN-3.5 · Full physiology — full-feature TabPFN.

Click a model row, or use **Model** at the top. For learned probability models, **Probability representation** switches between **Calibrated** and **Retained raw**; it changes which frozen output is displayed. Return to Calibrated to see the starting 30.4% with full TabPFN. **Held-out operation** lets you explore the other 21 cases afterward.

## 8. The actual result

These are the sealed TEST **raw Average Precision (AP)** results, not the single-case calibrated probabilities above:

| Model | Sealed TEST AP |
| --- | ---: |
| XGBoost, full physiology | **0.1858** |
| TabPFN-3.5, MAP only | **0.1839** |
| TabPFN-3.5, full physiology | **0.1740** |

**XGBoost achieved the strongest AP.** MAP-only TabPFN-3.5 used only **18 MAP-derived predictors** and came very close descriptively to full-physiology XGBoost's 74-feature representation. That does not establish statistical equivalence or superiority.

The point of the project is not that TabPFN universally beats XGBoost. The project tests whether continuous clinical physiology can be transformed into a rigorous causal tabular state that a foundation model can operate on.

## 9. Why Average Precision?

Only **116 of 3,146 TEST windows (about 3.7%)** are positive. Predicting “no event” almost everywhere could produce very high accuracy. AP is the primary metric because it focuses on ranking positives in this imbalanced setting.

## 10. What keeps the experiment honest?

- Predictors use data only at or before forecast time.
- No future interpolation or backward filling from future observations.
- No future outcomes in predictors or whole-operation normalization.
- Patient/case IDs are not model features.
- Patient groups are disjoint across training, tuning, calibration, and TEST.
- TEST was not used for model selection; calibration had its own partition.
- The website displays **frozen held-out predictions**. It makes no model calls.

## 11. How big was the study?

**150 patients / operations**, approximately **477.7 monitored hours**:

- 22,707 eligible prediction windows; 836 positive windows.
- 296 confirmed episodes; 247 evaluable episodes.
- Patients / operations: **90 training · 23 tuning · 15 calibration · 22 TEST**.
- Held-out TEST: **3,146 windows · 116 positives**.

## 12. Can I reproduce it?

The [scientific reproducibility documentation](../research/README.md) covers VitalDB ingestion, cohort construction, episode detection, anchors, labels/censoring, features, patient splitting, logistic regression, XGBoost, TabPFN-3.5, calibration, held-out evaluation, and replay export.

The [public replication documentation](../research/public_replication/README.md) provides a deterministic **12-case VitalDB replication mode** using the same causal processing pipeline and only public inputs. Hosted TabPFN requires separate authenticated access; its optional stage was tested offline and skipped in the public validation run.

**PUBLIC REPLICATION MODE: fully runnable**

**SEALED STUDY: partially reproducible**

The 12-case replication cohort did **not** produce the sealed AP values above. Private cohort/execution inputs and hosted-service variability limit exact sealed-study reproduction.

## 13. What are the limitations?

This is an **exploratory retrospective research prototype** from one public perioperative source and a relatively small cohort. It is not prospectively or externally validated, and it is not a clinically validated medical device.

**The main idea:** continuous operating-room signals → causal five-minute patient state → TabPFN-3.5 → probability of new sustained hypotension within the next five minutes.

[Live demo](https://intraop-hemodynamic-risk.pages.dev/demo) · [Main README](../README.md) · [Scientific reproducibility documentation](../research/README.md) · [Public replication documentation](../research/public_replication/README.md)
