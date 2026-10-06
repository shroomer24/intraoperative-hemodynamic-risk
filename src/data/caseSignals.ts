import { useEffect, useState } from 'react';
import { fetchReplayText, sha256 } from './assetIntegrity';
import { validateSignals } from './signalValidation';
import type { CaseSignals } from './signalValidation';
import type { CaseOrdinal, PresentationCase } from './types';

export function signalAssetPath(ordinal: string): string {
  if (!/^case-0(?:0[1-9]|1[0-9]|2[0-2])$/.test(ordinal)) throw new Error('Monitor traces unavailable');
  return `cases/${ordinal}/signals.json`;
}
export async function loadCaseSignals(selectedCase: PresentationCase, signal: AbortSignal): Promise<CaseSignals> {
  const path = signalAssetPath(selectedCase.ordinal);
  if (selectedCase.signalAsset.path !== path || !/^[a-f0-9]{64}$/.test(selectedCase.signalAsset.sha256)) {
    throw new Error('Monitor traces unavailable');
  }
  const text = await fetchReplayText(`/replay-v01/${path}`, signal);
  if (await sha256(text) !== selectedCase.signalAsset.sha256) throw new Error('Monitor traces unavailable');
  signal.throwIfAborted();
  return validateSignals(JSON.parse(text), selectedCase);
}

export type SignalLoadState = { ordinal: CaseOrdinal; status: 'loading' | 'unavailable' }
  | { ordinal: CaseOrdinal; status: 'ready'; data: CaseSignals };
export type SignalLoader = typeof loadCaseSignals;

export function useCaseSignals(selectedCase: PresentationCase, loader: SignalLoader = loadCaseSignals): SignalLoadState {
  const [state, setState] = useState<SignalLoadState>({ ordinal: selectedCase.ordinal, status: 'loading' });
  useEffect(() => {
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
  // Prevent even the render before the effect from pairing old traces with a new case label.
  return state.ordinal === selectedCase.ordinal ? state : { ordinal: selectedCase.ordinal, status: 'loading' };
}
