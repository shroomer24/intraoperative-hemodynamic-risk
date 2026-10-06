import type { IncomingMessage, ServerResponse } from 'node:http';
import { expect, it, vi } from 'vitest';
import { metadataPaths, rejectExcludedAsset } from '../tools/replay_asset_boundary';

it('defines exactly metadata + 22 signals + 22 forecasts + 22 events', () => {
  expect(metadataPaths).toHaveLength(68);
  expect(metadataPaths.filter(path => path.endsWith('/events.json'))).toHaveLength(22);
  expect(metadataPaths.filter(path => path.endsWith('/signals.json'))).toHaveLength(22);
  expect(metadataPaths.filter(path => path.endsWith('/prediction-windows.json'))).toHaveLength(22);
  expect(metadataPaths).not.toEqual(expect.arrayContaining([expect.stringMatching(/anchor-status/)]));
  for (const path of metadataPaths) {
    const response = { end: vi.fn() };
    expect(rejectExcludedAsset({ url: `/${path}` } as IncomingMessage, response as unknown as ServerResponse, '/approved-app/')).toBe(false);
    expect(response.end).not.toHaveBeenCalled();
  }
});
it.each([
  '/replay-v01/cases/case-004/anchor-status.json',
  '/replay-v01/cases/case-004/%61nchor-status.json',
  '/public/replay-v01/cases/case-004/prediction-windows.json',
  '/@fs/private/scientific-repository/data/table.csv',
  '/work/replay-private-v01/source_crosswalk.json',
  '/artifacts/modeling-v01/locked_execution/test/predictions.csv',
  '/data/raw/table.csv',
  '/intraop-prediction/pyproject.toml',
  '/replay-v01/cases/case-023/prediction-windows.json',
])('explicitly denies excluded/private route %s in development and preview', url => {
  const response = { statusCode: 200, end: vi.fn() };
  expect(rejectExcludedAsset({ url } as IncomingMessage, response as unknown as ServerResponse, '/approved-app/')).toBe(true);
  expect([403, 404]).toContain(response.statusCode);
  expect(response.end).toHaveBeenCalledOnce();
});
