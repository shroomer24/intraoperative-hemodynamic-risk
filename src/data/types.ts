export const modelIds = [
  'prevalence', 'current_map', 'logistic_map', 'logistic_full',
  'xgboost', 'tabpfn_map', 'tabpfn_full',
] as const;
export type ModelId = typeof modelIds[number];
export type CaseOrdinal = `case-${string}`;
export type ReplayMode = 'live' | 'review';

export interface ModelDefinition {
  id: ModelId;
  featureCount: number;
  representation: 'probability' | 'score';
  availableRepresentations: ('raw' | 'calibrated')[];
}

// Omits outcome counts, anchor-status pointers, labels and source identities.
export interface PresentationCase {
  ordinal: CaseOrdinal;
  durationSeconds: number;
  signalAsset: { path: string; sha256: string };
  predictionAsset: { path: string; sha256: string };
  eventAsset: { path: string; sha256: string };
}

export interface ShellMetadata {
  caseCount: number;
  predictionWindowCount: number;
  models: ModelDefinition[];
  cases: PresentationCase[];
  manifestHash: string;
  modelLockHash: string;
}

export const modelLabels: Record<ModelId, string> = {
  current_map: 'Current MAP',
  logistic_map: 'Logistic · MAP',
  logistic_full: 'Logistic · Full physiology',
  xgboost: 'XGBoost · Full physiology',
  tabpfn_map: 'TabPFN-3.5 · MAP',
  tabpfn_full: 'TabPFN-3.5 · Full physiology',
  prevalence: 'TRAINING prevalence',
};

export function caseLabel(ordinal: CaseOrdinal): string {
  return `Case ${ordinal.slice(5)}`;
}

export function formatDuration(seconds: number): string {
  const value = Math.floor(seconds);
  return [Math.floor(value / 3600), Math.floor(value / 60) % 60, value % 60]
    .map(part => String(part).padStart(2, '0')).join(':');
}
