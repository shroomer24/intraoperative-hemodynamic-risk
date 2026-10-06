import { act, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ReplayMetalMotion, useReplayMetalMotion, METAL_SETTLE_MS } from '../src/metal/ReplayMetalMotion';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
function Probe(){return <output data-testid="motion">{String(useReplayMetalMotion())}</output>;}
function harness(){const scheduler=new ManualScheduler();const store=new ReplayStore('case-019',2100,scheduler);store.activate();const view=render(<ReplayMetalMotion store={store}><Probe/></ReplayMetalMotion>);return {store,scheduler,...view};}
afterEach(()=>vi.useRealTimers());
describe('bounded decorative animation scheduling',()=>{
  it('is idle when paused, with no timer or replay frame',()=>{
    vi.useFakeTimers();const h=harness();expect(screen.getByTestId('motion').textContent).toBe('false');expect(vi.getTimerCount()).toBe(0);expect(h.scheduler.pending.size).toBe(0);h.store.dispose();
  });
  it('settles after pointer input even when the pointer remains over a control',()=>{
    vi.useFakeTimers();const h=harness();act(()=>window.dispatchEvent(new Event('pointermove')));expect(screen.getByTestId('motion').textContent).toBe('true');
    act(()=>vi.advanceTimersByTime(METAL_SETTLE_MS));expect(screen.getByTestId('motion').textContent).toBe('false');expect(vi.getTimerCount()).toBe(0);h.store.dispose();
  });
  it('runs with playback and stops on pause without changing replay position',()=>{
    const h=harness();h.store.seek(774);act(()=>h.store.play());expect(screen.getByTestId('motion').textContent).toBe('true');
    act(()=>h.scheduler.frame(100));act(()=>h.store.pause());expect(screen.getByTestId('motion').textContent).toBe('false');expect(h.store.getSnapshot().currentSeconds).toBe(777);expect(h.scheduler.pending.size).toBe(0);h.store.dispose();
  });
  it('bounds rapid input to one settle timer and cancels it on unmount',()=>{
    vi.useFakeTimers();const h=harness();act(()=>{for(let n=0;n<20;n++)window.dispatchEvent(new Event('keydown'));});expect(vi.getTimerCount()).toBe(1);h.unmount();expect(vi.getTimerCount()).toBe(0);h.store.dispose();
  });
});
