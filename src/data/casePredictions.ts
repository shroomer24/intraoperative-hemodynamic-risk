import { useEffect, useState } from 'react';
import { fetchReplayText, sha256 } from './assetIntegrity';
import { validatePredictions } from './predictionValidation';
import type { CasePredictions } from './predictionValidation';
import type { CaseOrdinal, PresentationCase } from './types';

export function predictionAssetPath(ordinal: string): string {
  if (!/^case-0(?:0[1-9]|1[0-9]|2[0-2])$/.test(ordinal)) throw new Error('Frozen forecasts unavailable.');
  return `cases/${ordinal}/prediction-windows.json`;
}
export async function loadCasePredictions(selectedCase: PresentationCase, signal: AbortSignal): Promise<CasePredictions> {
  const path = predictionAssetPath(selectedCase.ordinal);
  if (selectedCase.predictionAsset.path !== path || !/^[a-f0-9]{64}$/.test(selectedCase.predictionAsset.sha256)) {
    throw new Error('Frozen forecasts unavailable.');
  }
  const text = await fetchReplayText(`/replay-v01/${path}`, signal);
  if (await sha256(text) !== selectedCase.predictionAsset.sha256) throw new Error('Frozen forecasts unavailable.');
  signal.throwIfAborted();
  // Parsing transport bytes is necessary; the validator returns only forecast fields.
  return validatePredictions(JSON.parse(text), selectedCase);
}
export type PredictionLoadState = { ordinal?: CaseOrdinal; status: 'loading' | 'unavailable' }
  | { ordinal: CaseOrdinal; status: 'ready'; data: CasePredictions };
export type PredictionLoader = typeof loadCasePredictions;
export function useCasePredictions(selectedCase?: PresentationCase,
  loader: PredictionLoader = loadCasePredictions): PredictionLoadState {
  const [state, setState] = useState<PredictionLoadState>({ status: 'loading' });
  useEffect(() => {
    if (!selectedCase) return;
    const controller = new AbortController();
    let current = true;
    setState({ ordinal: selectedCase.ordinal, status: 'loading' });
    loader(selectedCase, controller.signal).then(data => {
      if (current && !controller.signal.aborted && data.ordinal === selectedCase.ordinal) {
        setState({ ordinal: selectedCase.ordinal, status: 'ready', data });
      }
    }).catch(() => {
      if (current && !controller.signal.aborted) setState({ ordinal: selectedCase.ordinal, status: 'unavailable' });
    });
    return () => { current = false; controller.abort(); };
  }, [selectedCase, loader]);
  // Also guards the render preceding effect cleanup when the selected case changes.
  return state.ordinal === selectedCase?.ordinal ? state : { ordinal: selectedCase?.ordinal, status: 'loading' };
}
