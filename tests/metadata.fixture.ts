import { readFileSync } from 'node:fs';
import { validateShellMetadata } from '../src/data/loader';

// These are the only scientific presentation files used by frontend tests.
export const manifestText = readFileSync('public/replay-v01/manifest.json', 'utf8');
export const indexText = readFileSync('public/replay-v01/cases/index.json', 'utf8');
export const manifest = JSON.parse(manifestText);
export const index = JSON.parse(indexText);
export const metadata = validateShellMetadata(manifest, index);
