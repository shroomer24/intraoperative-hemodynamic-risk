import { chooseOption } from './select.fixture';
import { readFileSync } from 'node:fs';
import { webcrypto } from 'node:crypto';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';
import { formatDuration, modelLabels } from '../src/data/types';
import { metadata } from './metadata.fixture';

beforeEach(() => {
  vi.stubGlobal('crypto', webcrypto);
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Offline shell test')));
});
afterEach(() => vi.unstubAllGlobals());
async function shell() {
  const result = render(<App loadData={() => Promise.resolve(metadata)} />);
  await screen.findByRole('heading', { name: 'Intraoperative Hemodynamic Risk' });
  await screen.findByRole('combobox', { name: 'Model' });
  return result;
}
describe('Phase 1 shell', () => {
  it('shows the approved title, subtitle and verified provenance', async () => {
    await shell();
    expect(screen.getByText('Historical held-out surgery · frozen predictions')).toBeTruthy();
    expect(screen.getByText('Frozen study')).toBeTruthy();
    expect(screen.getByText(/22 held-out operations/)).toBeTruthy();
    expect(screen.getByText(/3,146 eligible windows/)).toBeTruthy();
  });
  it('renders 22 safe case options without source identifiers or outcome counts', async () => {
    const { container } = await shell();
    fireEvent.click(screen.getByRole('combobox', { name: 'Held-out operation' }));
    const options = within(screen.getByRole('listbox')).getAllByRole('option');
    expect(options).toHaveLength(22);
    expect(options.map(option => option.firstChild?.textContent)).toEqual(metadata.cases.map((_, i) => `Case ${String(i + 1).padStart(3, '0')}`));
    expect(container.innerHTML).not.toMatch(/subject_id|case_id|positive_window_count|episode_count/);
  });
  it('keeps all model distinctions and a neutral default', async () => {
    await shell();
    const select = screen.getByRole('combobox', { name: 'Model' }) as HTMLButtonElement;
    expect(select.value).toBe('logistic_map');
    fireEvent.click(select);
    expect(within(screen.getByRole('listbox')).getAllByRole('option').map(option => option.firstChild?.textContent)).toEqual([
      modelLabels.current_map, modelLabels.logistic_map, modelLabels.logistic_full,
      modelLabels.xgboost, modelLabels.tabpfn_map, modelLabels.tabpfn_full, modelLabels.prevalence,
    ]);
    expect(select.textContent).not.toMatch(/Best|Winner|Recommended/);
  });
  it('uses score terminology exclusively in the Current MAP panel', async () => {
    await shell();
    chooseOption(screen.getByRole('combobox', { name: 'Model' }), 'current_map');
    const panel = screen.getByRole('region', { name: 'MAP-based ranking score' });
    expect(panel.textContent).toContain('Untransformed −latest MAP');
    expect(panel.textContent).not.toMatch(/probability|calibrated|%/i);
    expect(screen.getByRole('heading', { name: 'Score history' })).toBeTruthy();
  });
  it('uses probability terminology for every learned model', async () => {
    await shell();
    for (const value of ['logistic_map', 'logistic_full', 'xgboost', 'tabpfn_map', 'tabpfn_full']) {
      chooseOption(screen.getByRole('combobox', { name: 'Model' }), value);
      const panel = screen.getByRole('region', { name: 'Prediction' });
      expect(panel.textContent).toContain('Calibrated model estimate');
      expect(panel.textContent).toContain('Calibrated probability');
      expect(panel.textContent).toContain('No retained forecast at this time');
    }
  });
  it('represents prevalence as a raw TRAINING probability baseline', async () => {
    await shell();
    chooseOption(screen.getByRole('combobox', { name: 'Model' }), 'prevalence');
    const panel = screen.getByRole('region', { name: 'Prediction' });
    expect(panel.textContent).toContain('Fixed TRAINING probability baseline');
    expect(panel.textContent).toContain('Raw probability');
    expect(panel.textContent).not.toContain('Calibrated model estimate');
  });
  it('displays no fabricated percentages, vitals or model values', async () => {
    const { container } = await shell();
    expect(container.querySelector('.risk-readout')?.textContent).not.toMatch(/\d\s*%/);
    expect(screen.getByLabelText('No probability selected').textContent).toBe('—');
    for (const name of ['MAP', 'HR', 'SpO₂', 'ETCO₂']) {
      expect(screen.getByLabelText(new RegExp(`^${name},`)).textContent).toContain('—');
    }
    expect(container.querySelectorAll('canvas')).toHaveLength(0);
  });
  it('stores Live Replay and Review state without exposing outcomes', async () => {
    await shell();
    expect((screen.getByRole('radio', { name: 'Live Replay' }) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(screen.getByRole('radio', { name: 'Review' }));
    expect((screen.getByRole('radio', { name: 'Review' }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getAllByText('Historical future · not model input')[0]).toBeTruthy();
    fireEvent.click(screen.getByRole('radio', { name: 'Live Replay' }));
    expect(screen.getByText('Future physiology and outcomes hidden as the surgery progresses.')).toBeTruthy();
    expect(screen.getByLabelText('No probability selected').textContent).toBe('—');
  });
  it('changes the selected whole operation and real duration', async () => {
    await shell();
    chooseOption(screen.getByRole('combobox', { name: 'Held-out operation' }), 'case-002');
    expect(screen.getByText(formatDuration(metadata.cases[1].durationSeconds))).toBeTruthy();
    expect(screen.getByText('Case 002', { selector: '.operation-name' })).toBeTruthy();
  });
  it('random button selects one valid different whole operation', async () => {
    await shell();
    fireEvent.click(screen.getByRole('button', { name: 'Random Test Case' }));
    const selected = (screen.getByRole('combobox', { name: 'Held-out operation' }) as HTMLButtonElement).value;
    expect(selected).not.toBe('case-001');
    expect(metadata.cases.some(item => item.ordinal === selected)).toBe(true);
  });
  it('has a collapsed labelled comparison region and Escape returns focus', async () => {
    await shell();
    const button = screen.getByRole('button', { name: /Compare models/ });
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByRole('region', { name: 'Model comparison' })).toBeNull();
    fireEvent.click(button);
    expect(button.getAttribute('aria-expanded')).toBe('true');
    const region = screen.getByRole('region', { name: 'Model comparison' });
    expect(region.id).toBe(button.getAttribute('aria-controls'));
    expect(within(region).getAllByText(/aligned frozen forecasts/i)).toHaveLength(1);
    expect(region.querySelectorAll('.comparison-model-row')).toHaveLength(0);
    fireEvent.keyDown(region, { key: 'Escape' });
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(button);
  });
  it('keeps unavailable playback disabled and its timeline accessible', async () => {
    await shell();
    for (const name of [/Previous anchor/, /^Play$/, /Next anchor/]) {
      expect((screen.getByRole('button', { name }) as HTMLButtonElement).disabled).toBe(true);
    }
    expect((screen.getByRole('slider') as HTMLInputElement).disabled).toBe(true);
  });
  it('renders a safe failure state and retries metadata only', async () => {
    const loadData = vi.fn().mockRejectedValueOnce(new Error('private detail')).mockResolvedValue(metadata);
    render(<App loadData={loadData} />);
    expect((await screen.findByRole('alert')).textContent).not.toContain('private detail');
    fireEvent.click(screen.getByRole('button', { name: 'Retry metadata' }));
    await screen.findByRole('combobox', { name: 'Model' });
    expect(loadData).toHaveBeenCalledTimes(2);
  });
  it('supports reduced motion, visible focus and risk-first narrow layouts', () => {
    const css = readFileSync('src/styles/shell.css', 'utf8');
    expect(css).toContain('@media (prefers-reduced-motion: reduce)');
    expect(css).toContain('transition: none !important');
    expect(css).toContain(':focus-visible');
    expect(css).toMatch(/\.risk-panel\s*\{\s*order: -1/);
    expect(css).toMatch(/\[hidden\]\s*\{ display: none !important/);
  });
});
