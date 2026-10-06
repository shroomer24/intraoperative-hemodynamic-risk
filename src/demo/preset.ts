import type { CasePredictions } from '../data/predictionValidation';

// Verified against the existing approved Case 019 export; these are selection guards, not displayed values.
export const demoPreset = Object.freeze({ caseOrdinal: 'case-019' as const, modelId: 'tabpfn_full' as const,
  queryPosition: 2765, retainedCalibratedProbability: 0.3038048376358691 });
export function isDemoPath(path: string): boolean { return path === '/demo' || path === '/demo/'; }
export function resolveDemoAnchor(data: CasePredictions): number | undefined {
  if (data.ordinal !== demoPreset.caseOrdinal) return undefined;
  const matches = data.windows.filter(window => window.query_position === demoPreset.queryPosition);
  if (matches.length !== 1 || matches[0].model_outputs[demoPreset.modelId]?.calibrated_probability !== demoPreset.retainedCalibratedProbability) return undefined;
  return matches[0].anchor_seconds;
}
