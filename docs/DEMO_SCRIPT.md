# Judge demo

## About 30 seconds

Open `/demo`: Case 019, TabPFN-3.5 full physiology, Live Replay, calibrated 30.4% at the frozen 00:12:54 anchor.

“This replays a held-out surgery as if it were occurring live. The model sees only physiology available up to this point. This is its retained estimate of a new sustained hypotension episode beginning within five minutes.”

Press **Play**. At the default 30× speed, sustained confirmation appears about 3.3 seconds later. Pause after **00:14:32**, switch to **Review**, and expand **Compare models**.

“The historical episode began at 00:13:33 and was confirmed at 00:14:32. All displayed estimates are frozen outputs from the sealed evaluation; Review shows hindsight separately.”

## About 90 seconds

Open `/demo` and choose **10×** before pressing Play. Explain the safe case ordinal and historical replay. Point to the issued forecast, its original five-minute horizon, and the 30.4% calibrated estimate. The estimate stays attached to its issued anchor; it does not continuously refresh.

Press **Play**. Explain that Live Replay hides future traces and outcomes. Sustained confirmation appears about 9.8 seconds later at this speed. Pause after 00:14:32 and switch to **Review** to see the onset and confirmation annotations. Explain that these are historical outcomes, not model inputs.

Expand **Compare models**. Every row uses the same issued anchor and horizon. Select MAP-only and full-physiology models to compare retained estimates. Current MAP is a ranking score in mmHg, not a probability. TRAINING prevalence is a fixed baseline. Raw/calibrated selection changes only the preserved representation.

Conclude: “This research interface makes the timing and provenance of a sealed evaluation inspectable. It demonstrates historical model behavior, not clinical deployment readiness or a prospective intervention.”
