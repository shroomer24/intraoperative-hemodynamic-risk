import { readFileSync } from 'node:fs';
import { metadata } from './metadata.fixture';
import { validateEvents } from '../src/data/eventValidation';
import { validateForecastOutcomes } from '../src/data/forecastOutcomes';
import { predictionFixture } from './predictions.fixture';
export function outcomeFixture(ordinal = 'case-004') {
  const selectedCase = metadata.cases.find(item => item.ordinal === ordinal)!;
  const text = readFileSync(`public/replay-v01/${selectedCase.eventAsset.path}`, 'utf8');
  const rawEvents = JSON.parse(text);
  const prediction = predictionFixture(ordinal);
  return { ...prediction, eventText: text, rawEvents, events: validateEvents(rawEvents, selectedCase),
    outcomes: validateForecastOutcomes(prediction.raw, prediction.predictions, selectedCase) };
}
