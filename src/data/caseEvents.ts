import { useEffect, useState } from 'react';
import { fetchReplayText, sha256 } from './assetIntegrity';
import { validateEvents } from './eventValidation';
import type { CaseEvents } from './eventValidation';
import type { CaseOrdinal, PresentationCase } from './types';

export function eventAssetPath(ordinal: string): string {
  if (!/^case-0(?:0[1-9]|1[0-9]|2[0-2])$/.test(ordinal)) throw new Error('Historical events unavailable.');
  return `cases/${ordinal}/events.json`;
}
export async function loadCaseEvents(selectedCase: PresentationCase, signal: AbortSignal): Promise<CaseEvents> {
  const path = eventAssetPath(selectedCase.ordinal);
  if (selectedCase.eventAsset.path !== path || !/^[a-f0-9]{64}$/.test(selectedCase.eventAsset.sha256)) throw new Error('Historical events unavailable.');
  const text = await fetchReplayText(`/replay-v01/${path}`, signal);
  if (await sha256(text) !== selectedCase.eventAsset.sha256) throw new Error('Historical events unavailable.');
  signal.throwIfAborted(); return validateEvents(JSON.parse(text), selectedCase);
}
export type EventLoadState = { ordinal?: CaseOrdinal; status: 'loading' | 'unavailable' }
  | { ordinal: CaseOrdinal; status: 'ready'; data: CaseEvents };
export type EventLoader = typeof loadCaseEvents;
export function useCaseEvents(selectedCase?: PresentationCase, loader: EventLoader = loadCaseEvents): EventLoadState {
  const [state, setState] = useState<EventLoadState>({ status: 'loading' });
  useEffect(() => {
    if (!selectedCase) return;
    const controller = new AbortController(); let current = true;
    setState({ ordinal: selectedCase.ordinal, status: 'loading' });
    loader(selectedCase, controller.signal).then(data => {
      if (current && !controller.signal.aborted && data.ordinal === selectedCase.ordinal) setState({ ordinal: selectedCase.ordinal, status: 'ready', data });
    }).catch(() => {
      if (current && !controller.signal.aborted) setState({ ordinal: selectedCase.ordinal, status: 'unavailable' });
    });
    return () => { current = false; controller.abort(); };
  }, [selectedCase, loader]);
  return state.ordinal === selectedCase?.ordinal ? state : { ordinal: selectedCase?.ordinal, status: 'loading' };
}
