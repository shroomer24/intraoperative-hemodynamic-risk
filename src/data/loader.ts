import { modelIds } from './types';
import type { CaseOrdinal, ModelDefinition, PresentationCase, ShellMetadata } from './types';
import { fetchReplayText, sha256 } from './assetIntegrity';

export const MANIFEST_PATH = '/replay-v01/manifest.json';
export const CASE_INDEX_PATH = '/replay-v01/cases/index.json';
export const MANIFEST_HASH = '1cbeb72fc6211b946c3425fe38b794301c38cbb52cd6cc91c5796764efaa455c';
const LOCK_HASH = 'b2718a7c1ba448cc319c8594f93dc022480c573ef3dec3babd9b14dbdd021c0d';
const RECEIPT_HASH = '7ee5661490f1d81f3c442f638d2e9845e3d58136ee5112b3a494c52f2815b375';

function check(condition: unknown): asserts condition {
  if (!condition) throw new Error('Presentation metadata could not be verified.');
}
function object(value: unknown): Record<string, unknown> {
  check(value !== null && typeof value === 'object' && !Array.isArray(value));
  return value as Record<string, unknown>;
}

export function validateShellMetadata(manifestInput: unknown, indexInput: unknown): ShellMetadata {
  const manifest = object(manifestInput);
  check(manifest.schema_version === 'replay-v01' && manifest.case_count === 22);
  check(manifest.prediction_window_count === 3146);
  check(manifest.model_lock_sha256 === LOCK_HASH);
  check(manifest.test_completion_receipt_sha256 === RECEIPT_HASH);
  check(Array.isArray(manifest.models) && manifest.models.length === modelIds.length);
  const models = manifest.models.map((input): ModelDefinition => {
    const raw = object(input);
    check(modelIds.some(id => id === raw.id));
    const id = raw.id as ModelDefinition['id'];
    const expectedFeatures = { prevalence: 0, current_map: 1, logistic_map: 18,
      logistic_full: 74, xgboost: 74, tabpfn_map: 18, tabpfn_full: 74 }[id];
    check(raw.feature_count === expectedFeatures);
    const representation = id === 'current_map' ? 'score' : 'probability';
    check(raw.representation === representation);
    const representations: ('raw' | 'calibrated')[] =
      ['prevalence', 'current_map'].includes(id) ? ['raw'] : ['raw', 'calibrated'];
    check(JSON.stringify(raw.available_representations) === JSON.stringify(representations));
    return { id, featureCount: expectedFeatures, representation,
      availableRepresentations: representations };
  });
  check(new Set(models.map(model => model.id)).size === modelIds.length);
  check(Array.isArray(indexInput) && indexInput.length === manifest.case_count);
  const hashes = object(manifest.presentation_files_sha256);
  const cases = indexInput.map((input, i): PresentationCase => {
    const raw = object(input);
    const ordinal = `case-${String(i + 1).padStart(3, '0')}` as CaseOrdinal;
    check(raw.case_ordinal === ordinal);
    check(typeof raw.duration_seconds === 'number' && Number.isFinite(raw.duration_seconds));
    check(raw.duration_seconds > 0);
    const path = `cases/${ordinal}/signals.json`;
    check(object(raw.assets).signals === path);
    const sha256 = hashes[path];
    check(typeof sha256 === 'string' && /^[a-f0-9]{64}$/.test(sha256));
    const predictionPath = `cases/${ordinal}/prediction-windows.json`;
    check(object(raw.assets)['prediction-windows'] === predictionPath);
    const predictionHash = hashes[predictionPath];
    check(typeof predictionHash === 'string' && /^[a-f0-9]{64}$/.test(predictionHash));
    const eventPath = `cases/${ordinal}/events.json`;
    check(object(raw.assets).events === eventPath);
    const eventHash = hashes[eventPath];
    check(typeof eventHash === 'string' && /^[a-f0-9]{64}$/.test(eventHash));
    return { ordinal, durationSeconds: raw.duration_seconds, signalAsset: { path, sha256 },
      predictionAsset: { path: predictionPath, sha256: predictionHash }, eventAsset: { path: eventPath, sha256: eventHash } };
  });
  return { caseCount: manifest.case_count, predictionWindowCount: manifest.prediction_window_count,
    models, cases, manifestHash: MANIFEST_HASH, modelLockHash: LOCK_HASH };
}

export async function loadShellMetadata(): Promise<ShellMetadata> {
  const manifestText = await fetchReplayText(MANIFEST_PATH);
  check(await sha256(manifestText) === MANIFEST_HASH);
  const manifest = object(JSON.parse(manifestText));
  const indexHash = object(manifest.presentation_files_sha256)['cases/index.json'];
  check(typeof indexHash === 'string');
  const indexText = await fetchReplayText(CASE_INDEX_PATH);
  check(await sha256(indexText) === indexHash);
  return validateShellMetadata(manifest, JSON.parse(indexText));
}

export function randomCase(cases: PresentationCase[], current: CaseOrdinal): CaseOrdinal {
  check(cases.length > 0);
  const pool = cases.length > 1 ? cases.filter(item => item.ordinal !== current) : cases;
  const max = 0x100000000;
  const limit = max - max % pool.length;
  const sample = new Uint32Array(1);
  do { crypto.getRandomValues(sample); } while (sample[0] >= limit);
  return pool[sample[0] % pool.length].ordinal;
}
