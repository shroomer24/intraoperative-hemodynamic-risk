import { readFileSync } from 'node:fs';
import { webcrypto } from 'node:crypto';
import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';
import { CaseContext, contextLine } from '../src/case-context/CaseContext';
import { caseContexts } from '../src/case-context/records';
import type { CaseContextRecord } from '../src/case-context/records';
import { metadata } from './metadata.fixture';
import { chooseOption } from './select.fixture';

beforeEach(() => {
  history.replaceState(null, '', '/');
  vi.stubGlobal('crypto', webcrypto);
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Offline context test')));
});
afterEach(() => { history.replaceState(null, '', '/'); vi.unstubAllGlobals(); });
const record = (procedure: string, department: string, approach: string): CaseContextRecord => ({
  case_ordinal: 'case-001', procedure, department, approach,
});
async function shell() {
  render(<App loadData={() => Promise.resolve(metadata)} />);
  await screen.findByRole('combobox', { name: 'Held-out operation' });
}
function expectCurrentContext() {
  const ordinal = (screen.getByRole('combobox', { name: 'Held-out operation' }) as HTMLButtonElement).value;
  const element = document.querySelector('.case-context');
  expect(element?.getAttribute('data-case-ordinal')).toBe(ordinal);
  expect(element?.textContent).toBe(contextLine(caseContexts.find(item => item.case_ordinal === ordinal)));
  expect(document.querySelectorAll('.case-context')).toHaveLength(1);
}

describe('Verified presentation-only case context', () => {
  it('matches the reviewed safe table, with exactly one four-field record for all 22 public ordinals', () => {
    const reviewed = JSON.parse(readFileSync('tests/fixtures/case-context-v01.json', 'utf8'));
    expect(caseContexts).toEqual(reviewed);
    expect(caseContexts.map(item => item.case_ordinal)).toEqual(metadata.cases.map(item => item.ordinal));
    expect(new Set(caseContexts.map(item => item.case_ordinal)).size).toBe(22);
    for (const item of caseContexts) expect(Object.keys(item).sort()).toEqual(['approach', 'case_ordinal', 'department', 'procedure']);
    expect(JSON.stringify(caseContexts)).not.toMatch(/source_case|subject_id|diagnosis|opstart|opend|\/Users\/|\.csv/);
  });
  it('omits unavailable approach without an empty separator', () => {
    expect(contextLine(record('Metastasectomy', 'Thoracic surgery', 'Not available'))).toBe('Metastasectomy · Thoracic surgery');
    expect(contextLine(record('Metastasectomy', 'Thoracic surgery', '  '))).toBe('Metastasectomy · Thoracic surgery');
  });
  it('shows department only when procedure is unavailable, even if approach exists', () => {
    expect(contextLine(record('Not available', 'Thoracic surgery', 'Videoscopic'))).toBe('Thoracic surgery');
  });
  it('shows the explicit fallback when neither procedure nor department is available', () => {
    expect(contextLine(undefined)).toBe('Surgical context not available');
    expect(contextLine(record('', 'Not available', 'Videoscopic'))).toBe('Surgical context not available');
  });
  it('rerenders all 22 ordinals synchronously without carrying prior context', () => {
    const view = render(<CaseContext ordinal="case-001" />);
    for (const item of [...caseContexts, ...[...caseContexts].reverse()]) {
      view.rerender(<CaseContext ordinal={item.case_ordinal} />);
      expect(screen.getByLabelText(`Surgical context for Case ${item.case_ordinal.slice(-3)}`).textContent).toBe(contextLine(item));
    }
  });
  it('tracks rapid shell selection, including 001, 019 and 011', async () => {
    await shell();
    expectCurrentContext();
    for (const ordinal of ['case-019', 'case-011', 'case-001', ...caseContexts.map(item => item.case_ordinal)]) {
      chooseOption(screen.getByRole('combobox', { name: 'Held-out operation' }), ordinal);
      expectCurrentContext();
    }
  });
  it('tracks the actual random selected case', async () => {
    await shell();
    for (let i = 0; i < 5; i++) {
      fireEvent.click(screen.getByRole('button', { name: 'Random Test Case' }));
      expectCurrentContext();
    }
  });
  it('starts /demo on the verified Case 019 context', async () => {
    history.replaceState(null, '', '/demo');
    await shell();
    expectCurrentContext();
    expect(screen.getByLabelText('Surgical context for Case 019').textContent).toBe('Metastasectomy · Thoracic surgery · Videoscopic');
  });
  it('bounds visual text to two lines without another duration or metadata card', () => {
    const css = readFileSync('src/styles/case-context.css', 'utf8');
    expect(css).toContain('-webkit-line-clamp: 2');
    expect(css).toContain('overflow-wrap: anywhere');
    const view = render(<CaseContext ordinal="case-019" />);
    expect(view.container.querySelectorAll('p')).toHaveLength(1);
    expect(view.container.textContent).not.toMatch(/35 min|2100/);
  });
});
