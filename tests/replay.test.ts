import { describe, expect, it, vi } from 'vitest';
import { ReplayStore } from '../src/replay/store';
import { ReplayClock } from '../src/replay/clock';
import { ManualScheduler } from './replay.fixture';
import { exactSample, replayViewport, visibleWindow } from '../src/replay/viewport';
import { pauseWhenHidden, replayKeyboard } from '../src/replay/visibility';
import { timeSplits } from '../src/charts/synchronizedAxes';
import { signalFixture } from './signals.fixture';

function session(duration = 7500) {
  const scheduler = new ManualScheduler();
  const store = new ReplayStore('case-004', duration, scheduler);
  return { scheduler, store };
}
const fixture = signalFixture();

describe('one deterministic presentation clock', () => {
  it('starts at zero, paused, Live, 30×, first fifteen minutes, awaiting verification', () => {
    const { store, scheduler } = session();
    expect(store.getSnapshot()).toEqual({ caseOrdinal: 'case-004', durationSeconds: 7500,
      currentSeconds: 0, isPlaying: false, playbackSpeed: 30, replayMode: 'live', visibleWindow: [0, 900], ready: false });
    store.play(); expect(scheduler.pending.size).toBe(0);
    store.activate(); expect(store.getSnapshot().currentSeconds).toBe(0);
  });
  it('derives time from monotonic elapsed wall time, even when frames arrive irregularly', () => {
    const { store, scheduler } = session(); store.activate(); store.play();
    scheduler.frame(120); expect(store.getSnapshot().currentSeconds).toBe(3.6);
    scheduler.frame(880); expect(store.getSnapshot().currentSeconds).toBe(30);
    scheduler.frame(2000); expect(store.getSnapshot().currentSeconds).toBe(90);
    store.pause(); expect(scheduler.pending.size).toBe(0);
    scheduler.frame(100000); expect(store.getSnapshot().currentSeconds).toBe(90);
    store.play(); scheduler.frame(1000); expect(store.getSnapshot().currentSeconds).toBe(120);
    store.dispose();
  });
  it.each([1, 10, 30, 60] as const)('preserves exact position while switching to %i× and uses the new speed', speed => {
    const { store, scheduler } = session(); store.activate(); store.play();
    scheduler.wall = 1234; store.setSpeed(speed);
    const position = store.getSnapshot().currentSeconds;
    expect(position).toBeCloseTo(37.02);
    scheduler.frame(1000); expect(store.getSnapshot().currentSeconds).toBeCloseTo(position + speed);
    store.dispose();
  });
  it('clamps at duration and automatically pauses without another animation frame', () => {
    const { store, scheduler } = session(10); store.activate(); store.play(); scheduler.frame(1000);
    expect(store.getSnapshot().currentSeconds).toBe(10); expect(store.getSnapshot().isPlaying).toBe(false);
    expect(scheduler.pending.size).toBe(0); store.play(); expect(scheduler.pending.size).toBe(0);
    store.seek(9999); expect(store.getSnapshot().currentSeconds).toBe(10);
    store.seek(-8); expect(store.getSnapshot().currentSeconds).toBe(0);
  });
  it('scrubs immediately, pauses, cancels scheduling and never keeps a catch-up anchor', () => {
    const { store, scheduler } = session(); store.activate(); store.play(); scheduler.frame(1000);
    store.seek(2345); expect(store.getSnapshot().currentSeconds).toBe(2345);
    expect(store.getSnapshot().visibleWindow).toEqual([1445, 2345]);
    expect(store.getSnapshot().isPlaying).toBe(false); expect(scheduler.pending.size).toBe(0);
    scheduler.frame(10000); store.play(); scheduler.frame(1000);
    expect(store.getSnapshot().currentSeconds).toBe(2375); store.dispose();
  });
  it('pauses on document.hidden, blocks hidden play, and never catches up on foreground', () => {
    const { store, scheduler } = session(); store.activate();
    const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
    const cleanup = pauseWhenHidden(store); store.play(); scheduler.frame(1000);
    hidden.mockReturnValue(true); document.dispatchEvent(new Event('visibilitychange'));
    expect(store.getSnapshot().isPlaying).toBe(false); expect(scheduler.pending.size).toBe(0);
    scheduler.frame(60000); store.play(); expect(scheduler.pending.size).toBe(0);
    hidden.mockReturnValue(false); document.dispatchEvent(new Event('visibilitychange'));
    expect(store.getSnapshot().currentSeconds).toBe(30); expect(store.getSnapshot().isPlaying).toBe(false);
    store.play(); scheduler.frame(1000); expect(store.getSnapshot().currentSeconds).toBe(60);
    cleanup(); store.dispose(); hidden.mockRestore();
  });
  it('case replacement resets all state and disposal cancels the former clock', () => {
    const { store, scheduler } = session(); store.activate(); store.seek(2345); store.setSpeed(60); store.setMode('review'); store.play();
    store.dispose(); const next = new ReplayStore('case-019', 2100, scheduler); next.activate();
    expect(next.getSnapshot()).toMatchObject({ caseOrdinal: 'case-019', currentSeconds: 0, isPlaying: false, replayMode: 'live', playbackSpeed: 30, visibleWindow: [0, 900] });
    scheduler.frame(10000); expect(next.getSnapshot().currentSeconds).toBe(0);
  });
  it('publishes at most 10 times per wall second while delivering one shared imperative frame to all four cursors', () => {
    const { store, scheduler } = session(); store.activate(); store.play();
    const cursors = Array.from({ length: 4 }, () => [] as number[]);
    const cleanups = cursors.map(cursor => store.subscribeFrame(time => cursor.push(time)));
    const before = store.diagnostics.publications;
    for (let i = 0; i < 625; i++) scheduler.frame(16);
    expect(store.diagnostics.publications - before).toBeLessThanOrEqual(100);
    expect(store.diagnostics.frames).toBe(625);
    for (const cursor of cursors) expect(cursor).toEqual(cursors[0]);
    store.pause(); const idle = { ...store.diagnostics };
    for (let i = 0; i < 100; i++) scheduler.frame(16);
    expect(store.diagnostics).toEqual(idle); expect(scheduler.pending.size).toBe(0);
    cleanups.forEach(cleanup => cleanup());
  });
  it('manual playback works regardless of reduced-motion preferences; there is no autoplay', () => {
    const { store, scheduler } = session(); store.activate();
    expect(scheduler.pending.size).toBe(0); store.play(); scheduler.frame(1000);
    expect(store.getSnapshot().currentSeconds).toBe(30); store.dispose();
  });
  it('clock pause and seek retain fractions without accumulated timer increments', () => {
    let wall = 0; const clock = new ReplayClock(100, () => wall); clock.play(); wall = 17;
    clock.pause(); expect(clock.position()).toBe(0.51); wall += 90000;
    expect(clock.position()).toBe(0.51); clock.seek(10.125); clock.play(); wall += 100;
    expect(clock.position()).toBeCloseTo(13.125);
  });
});

describe('immutable exact-grid presentation selectors', () => {
  it('holds the first domain through 900 and rolls precisely thereafter, bounded at end', () => {
    expect(visibleWindow(0, 300)).toEqual([0, 300]);
    expect(visibleWindow(899.9, 7500)).toEqual([0, 900]);
    expect(visibleWindow(900, 7500)).toEqual([0, 900]);
    expect(visibleWindow(900.25, 7500)).toEqual([0.25, 900.25]);
    expect(visibleWindow(99999, 2100)).toEqual([1200, 2100]);
    expect(timeSplits([14400, 15300], 800)).toEqual([14400, 14580, 14760, 14940, 15120, 15300]);
  });
  it('Live has no future sample in any series or scale input, but Review exposes exact history', () => {
    const domain = visibleWindow(81.9, 7500);
    const live = replayViewport(fixture.signals, 81.9, 'live', domain);
    expect(live.timeSeconds.at(-1)).toBe(81); expect(live.timeSeconds).toHaveLength(82);
    const review = replayViewport(fixture.signals, 81.9, 'review', domain);
    expect(review.timeSeconds.at(-1)).toBe(900);
    for (const id of ['map', 'hr', 'spo2', 'etco2'] as const) {
      expect(live.values[id]).toEqual(fixture.signals.channels[id].values.slice(0, 82));
      expect(review.values[id]).toEqual(fixture.signals.channels[id].values.slice(0, 901));
    }
  });
  it('current lookup floors the exact second and preserves null instead of forward/backward filling', () => {
    expect(exactSample(fixture.signals, 'map', 80.99)).toBe(fixture.signals.channels.map.values[80]);
    expect(exactSample(fixture.signals, 'map', 81.99)).toBeNull();
    expect(exactSample(fixture.signals, 'map', 91.99)).toBeNull();
    expect(exactSample(fixture.signals, 'map', 92)).toBe(fixture.signals.channels.map.values[92]);
    expect(fixture.signals.channels.map.values[80]).not.toBeNull();
    expect(fixture.signals.channels.map.values[92]).not.toBeNull();
  });
  it('keeps gaps and source arrays unchanged, with bounded slices throughout the long operation', () => {
    const long = signalFixture('case-010').signals;
    const before = JSON.stringify(long);
    const identities = [long.timeSeconds, ...Object.values(long.channels).map(channel => channel.values)];
    for (let time = 0; time <= 30900; time += 17) {
      const view = replayViewport(long, time, 'live', visibleWindow(time, 30900));
      expect(view.timeSeconds.length).toBeLessThanOrEqual(901);
      expect(view.timeSeconds.at(-1)).toBe(time);
    }
    expect(JSON.stringify(long)).toBe(before);
    expect([long.timeSeconds, ...Object.values(long.channels).map(channel => channel.values)]).toEqual(identities);
    const view = replayViewport(fixture.signals, 160, 'live', [0, 900]);
    expect(view.values.map[81]).toBeNull(); expect(view.values.map[151]).toBeNull();
  });
});

describe('keyboard ownership', () => {
  it('supports global shortcuts and shift steps while leaving every native/editable control alone', () => {
    const { store, scheduler } = session(); store.activate(); const cleanup = replayKeyboard(store);
    const key = (target: EventTarget, key: string, shiftKey = false) => target.dispatchEvent(new KeyboardEvent('keydown', { key, shiftKey, bubbles: true, cancelable: true }));
    key(document.body, ' '); expect(store.getSnapshot().isPlaying).toBe(true);
    key(document.body, ' '); expect(store.getSnapshot().isPlaying).toBe(false);
    key(document.body, 'ArrowRight'); expect(store.getSnapshot().currentSeconds).toBe(1);
    key(document.body, 'ArrowRight', true); expect(store.getSnapshot().currentSeconds).toBe(11);
    key(document.body, 'ArrowLeft', true); expect(store.getSnapshot().currentSeconds).toBe(1);
    key(document.body, 'End'); expect(store.getSnapshot().currentSeconds).toBe(7500);
    key(document.body, 'Home'); expect(store.getSnapshot().currentSeconds).toBe(0);
    for (const tag of ['input', 'select', 'textarea', 'button', 'div']) {
      const control = document.createElement(tag); if (tag === 'div') control.setAttribute('contenteditable', 'true');
      document.body.append(control); key(control, 'End'); key(control, ' ');
      expect(store.getSnapshot().currentSeconds).toBe(0); expect(scheduler.pending.size).toBe(0); control.remove();
    }
    cleanup(); key(document.body, 'End'); expect(store.getSnapshot().currentSeconds).toBe(0);
  });
});
