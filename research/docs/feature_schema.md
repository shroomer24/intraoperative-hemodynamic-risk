# Exact v0.1 feature schema (74 ordered columns)

Schema order is the FEATURE_NAMES tuple in src/features/core.py and is exported
unchanged to feature_schema.json. Six fixed signals are MAP, SBP, DBP, HR, SpO2,
and **Solar8000/ETCO2**. Units below follow the frozen release track dictionary.
No Primus fallback, static variables, identifiers, future/censoring fields, or
waveform-derived predictors occur in this schema.

All feature windows use (t-300,t]; recent summaries use (t-60,t]. Summaries use
valid causal one-second held values; median/quantiles ignore missing seconds.
Quantiles use NumPy's linear method; standard deviation uses ddof=0. A summary
needs at least one valid value, otherwise it is NaN. Latest means the latest
valid held value anywhere in the history, not a fabricated current value when
currently stale. Current staleness is conveyed by missing fraction and age.

Base-signal slopes fit the **actual original valid measurement timestamps** in
the corresponding window, with at least two distinct measurement times. They
never treat repeated held grid values as independent original observations.
Duplicate measurement times use the last valid original record. Two is the
minimum mathematically identifiable OLS sample count, an explicit v0.1 technical
choice; it is not a clinical reliability claim. Constant actual values give zero
slope. No whole-case centering/scaling is applied.

Pulse pressure is computed only at grid seconds where both held SBP and DBP
are valid; their ages are <=10 seconds, so alignment is bounded. Its slope fits
the aligned integer-second values, uses real elapsed seconds rather than the
positions of nonmissing rows, and additionally requires at least two actual
SBP measurements and two actual DBP measurements in the history. HR/SBP uses
valid aligned HR and **positive** SBP. Undefined/nonfinite ratios become missing;
no diagnostic interpretation or additional physiological ranges are imposed.

Minute 1 is the oldest: (t-300,t-240]; minute 5 is latest: (t-60,t]. The change
feature subtracts minute 4 from minute 5. MAP fractions use only valid historical
MAP seconds as denominator and strict <70/<75 thresholds. Thus values exactly
70/75 are excluded from their respective below-threshold fractions. Keep gray-zone
65–75 mmHg observations; they are not automatically discarded.

Missing fraction is invalid/stale grid seconds /300. Measurement age is t minus
the latest valid recording observation available by t, even if now older than ten
seconds; NaN means no such observation has arrived. Age may exceed 300 seconds;
no old physiological value is made current by retaining its age. Elapsed seconds
is t from recording zero, never a surgical duration or end-time predictor.

| Order | Feature | Unit | Definition |
|---:|---|---|---|
| 1 | `map_latest` | mmHg | latest |
| 2 | `map_median_60s` | mmHg | median 60s |
| 3 | `map_median_300s` | mmHg | median 300s |
| 4 | `map_std_300s` | mmHg | std 300s |
| 5 | `map_p10_300s` | mmHg | p10 300s |
| 6 | `map_p90_300s` | mmHg | p90 300s |
| 7 | `map_slope_60s` | mmHg/s | slope 60s |
| 8 | `map_slope_300s` | mmHg/s | slope 300s |
| 9 | `sbp_latest` | mmHg | latest |
| 10 | `sbp_median_60s` | mmHg | median 60s |
| 11 | `sbp_median_300s` | mmHg | median 300s |
| 12 | `sbp_std_300s` | mmHg | std 300s |
| 13 | `sbp_p10_300s` | mmHg | p10 300s |
| 14 | `sbp_p90_300s` | mmHg | p90 300s |
| 15 | `sbp_slope_60s` | mmHg/s | slope 60s |
| 16 | `sbp_slope_300s` | mmHg/s | slope 300s |
| 17 | `dbp_latest` | mmHg | latest |
| 18 | `dbp_median_60s` | mmHg | median 60s |
| 19 | `dbp_median_300s` | mmHg | median 300s |
| 20 | `dbp_std_300s` | mmHg | std 300s |
| 21 | `dbp_p10_300s` | mmHg | p10 300s |
| 22 | `dbp_p90_300s` | mmHg | p90 300s |
| 23 | `dbp_slope_60s` | mmHg/s | slope 60s |
| 24 | `dbp_slope_300s` | mmHg/s | slope 300s |
| 25 | `hr_latest` | /min | latest |
| 26 | `hr_median_60s` | /min | median 60s |
| 27 | `hr_median_300s` | /min | median 300s |
| 28 | `hr_std_300s` | /min | std 300s |
| 29 | `hr_p10_300s` | /min | p10 300s |
| 30 | `hr_p90_300s` | /min | p90 300s |
| 31 | `hr_slope_60s` | /min/s | slope 60s |
| 32 | `hr_slope_300s` | /min/s | slope 300s |
| 33 | `spo2_latest` | % | latest |
| 34 | `spo2_median_60s` | % | median 60s |
| 35 | `spo2_median_300s` | % | median 300s |
| 36 | `spo2_std_300s` | % | std 300s |
| 37 | `spo2_p10_300s` | % | p10 300s |
| 38 | `spo2_p90_300s` | % | p90 300s |
| 39 | `spo2_slope_60s` | %/s | slope 60s |
| 40 | `spo2_slope_300s` | %/s | slope 300s |
| 41 | `etco2_latest` | mmHg | latest |
| 42 | `etco2_median_60s` | mmHg | median 60s |
| 43 | `etco2_median_300s` | mmHg | median 300s |
| 44 | `etco2_std_300s` | mmHg | std 300s |
| 45 | `etco2_p10_300s` | mmHg | p10 300s |
| 46 | `etco2_p90_300s` | mmHg | p90 300s |
| 47 | `etco2_slope_60s` | mmHg/s | slope 60s |
| 48 | `etco2_slope_300s` | mmHg/s | slope 300s |
| 49 | `map_median_latest_minus_previous_minute` | mmHg | Minute 5 median minus minute 4 median |
| 50 | `map_minute_1_median` | mmHg | Ordered non-overlapping minute median (1 oldest, 5 latest) |
| 51 | `map_minute_2_median` | mmHg | Ordered non-overlapping minute median (1 oldest, 5 latest) |
| 52 | `map_minute_3_median` | mmHg | Ordered non-overlapping minute median (1 oldest, 5 latest) |
| 53 | `map_minute_4_median` | mmHg | Ordered non-overlapping minute median (1 oldest, 5 latest) |
| 54 | `map_minute_5_median` | mmHg | Ordered non-overlapping minute median (1 oldest, 5 latest) |
| 55 | `map_fraction_below_70` | dimensionless | Fraction of valid MAP seconds strictly below threshold |
| 56 | `map_fraction_below_75` | dimensionless | Fraction of valid MAP seconds strictly below threshold |
| 57 | `pulse_pressure_latest` | mmHg | Aligned SBP minus DBP; latest |
| 58 | `pulse_pressure_median_300s` | mmHg | Aligned SBP minus DBP; median_300s |
| 59 | `pulse_pressure_slope_300s` | mmHg/s | Aligned SBP minus DBP; slope_300s |
| 60 | `hr_sbp_ratio_latest` | (/min)/mmHg | Aligned HR divided by positive SBP; latest |
| 61 | `hr_sbp_ratio_median_60s` | (/min)/mmHg | Aligned HR divided by positive SBP; median_60s |
| 62 | `map_missing_fraction` | dimensionless | Missing/stale seconds divided by 300 |
| 63 | `map_measurement_age_seconds` | s | Age of latest valid recording measurement available by t |
| 64 | `sbp_missing_fraction` | dimensionless | Missing/stale seconds divided by 300 |
| 65 | `sbp_measurement_age_seconds` | s | Age of latest valid recording measurement available by t |
| 66 | `dbp_missing_fraction` | dimensionless | Missing/stale seconds divided by 300 |
| 67 | `dbp_measurement_age_seconds` | s | Age of latest valid recording measurement available by t |
| 68 | `hr_missing_fraction` | dimensionless | Missing/stale seconds divided by 300 |
| 69 | `hr_measurement_age_seconds` | s | Age of latest valid recording measurement available by t |
| 70 | `spo2_missing_fraction` | dimensionless | Missing/stale seconds divided by 300 |
| 71 | `spo2_measurement_age_seconds` | s | Age of latest valid recording measurement available by t |
| 72 | `etco2_missing_fraction` | dimensionless | Missing/stale seconds divided by 300 |
| 73 | `etco2_measurement_age_seconds` | s | Age of latest valid recording measurement available by t |
| 74 | `elapsed_seconds` | s | Anchor time since recording start |
