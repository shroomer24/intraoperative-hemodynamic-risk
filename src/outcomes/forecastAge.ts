import type { PredictionWindow } from '../data/predictionValidation';
export function forecastAge(forecast: PredictionWindow | undefined, current: number) {
  if (!forecast) return undefined;
  const end = forecast.horizon_end_inclusive_seconds;
  if (current <= end) return { expired: false, title: 'Forecast age', value: `${Math.floor(current) - forecast.anchor_seconds} s` };
  const elapsed = Math.floor(current - end);
  return { expired: true, title: 'Forecast horizon ended', value: `${Math.floor(elapsed / 60)}m ${String(elapsed % 60).padStart(2, '0')}s ago` };
}
