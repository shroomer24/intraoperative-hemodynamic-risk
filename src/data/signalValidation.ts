import type { CaseOrdinal, PresentationCase } from './types';

export const channelIds = ['map', 'hr', 'spo2', 'etco2', 'sbp', 'dbp'] as const;
export type ChannelId = typeof channelIds[number];
export type DisplayChannelId = 'map' | 'hr' | 'spo2' | 'etco2';
export type Observations = (number | null)[];
export interface SignalChannel { unit: string; values: Observations; age_seconds: Observations }
export interface CaseSignals {
  ordinal: CaseOrdinal;
  timeSeconds: number[];
  channels: Record<ChannelId, SignalChannel>;
}
function check(condition: unknown): asserts condition {
  if (!condition) throw new Error('Monitor traces unavailable');
}
function record(value: unknown, keys: readonly string[]): Record<string, unknown> {
  check(value !== null && typeof value === 'object' && !Array.isArray(value));
  check(Object.keys(value).sort().join('|') === [...keys].sort().join('|'));
  return value as Record<string, unknown>;
}

export function validateSignals(input: unknown, selectedCase: PresentationCase): CaseSignals {
  const root = record(input, ['time_seconds', 'channels']);
  const time = root.time_seconds;
  check(Array.isArray(time) && time.length === Math.floor(selectedCase.durationSeconds) + 1);
  // Frozen export: surgical-relative, strictly increasing, one-second grid starting at zero.
  check(time.every((value, i) => typeof value === 'number' && Number.isFinite(value) && value === i));
  const rawChannels = record(root.channels, channelIds);
  const units = { map: 'mmHg', hr: 'beats/min', spo2: '%', etco2: 'mmHg', sbp: 'mmHg', dbp: 'mmHg' };
  const channels = {} as Record<ChannelId, SignalChannel>;
  for (const id of channelIds) {
    const channel = record(rawChannels[id], ['unit', 'values', 'age_seconds']);
    check(channel.unit === units[id]);
    for (const name of ['values', 'age_seconds'] as const) {
      const values = channel[name];
      check(Array.isArray(values) && values.length === time.length);
      check(values.every(value => value === null || (typeof value === 'number'
        && Number.isFinite(value) && (name !== 'age_seconds' || value >= 0))));
    }
    // Preserve source arrays and nulls. Age is validated but never reapplied to values.
    channels[id] = channel as unknown as SignalChannel;
  }
  return { ordinal: selectedCase.ordinal, timeSeconds: time, channels };
}
