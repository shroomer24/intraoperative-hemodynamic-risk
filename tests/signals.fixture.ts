import { readFileSync } from 'node:fs';
import { metadata } from './metadata.fixture';
import { validateSignals } from '../src/data/signalValidation';

export function signalFixture(ordinal = 'case-004') {
  const selectedCase = metadata.cases.find(item => item.ordinal === ordinal)!;
  const text = readFileSync(`public/replay-v01/${selectedCase.signalAsset.path}`, 'utf8');
  const raw = JSON.parse(text);
  return { selectedCase, text, raw, signals: validateSignals(raw, selectedCase) };
}
