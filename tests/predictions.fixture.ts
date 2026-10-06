import { readFileSync } from 'node:fs';
import { metadata } from './metadata.fixture';
import { validatePredictions } from '../src/data/predictionValidation';

export function predictionFixture(ordinal = 'case-004') {
  const selectedCase = metadata.cases.find(item => item.ordinal === ordinal)!;
  const text = readFileSync(`public/replay-v01/${selectedCase.predictionAsset.path}`, 'utf8');
  const raw = JSON.parse(text);
  // Only this forecast-only result may enter runtime selectors/UI.
  return { selectedCase, text, raw, predictions: validatePredictions(raw, selectedCase) };
}
