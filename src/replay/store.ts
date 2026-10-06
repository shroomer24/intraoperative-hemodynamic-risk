import { useSyncExternalStore } from 'react';
import type { CaseOrdinal, ReplayMode } from '../data/types';
import { ReplayClock, browserScheduler, playbackSpeeds } from './clock';
import type { FrameScheduler, PlaybackSpeed } from './clock';
import { visibleWindow } from './viewport';

export interface ReplayState {
  caseOrdinal: CaseOrdinal;
  currentSeconds: number;
  durationSeconds: number;
  isPlaying: boolean;
  playbackSpeed: PlaybackSpeed;
  replayMode: ReplayMode;
  visibleWindow: readonly [number, number];
  ready: boolean;
}
export const PUBLICATION_INTERVAL_MS = 100;
export class ReplayStore {
  private readonly clock: ReplayClock;
  private state: ReplayState;
  private readonly listeners = new Set<() => void>();
  private readonly frames = new Set<(current: number) => void>();
  private hidden = false;
  private chartDomain: readonly [number, number] | undefined;
  private handle: number | undefined;
  private lastPublished = -Infinity;
  readonly diagnostics = { frames: 0, publications: 0 };
  constructor(caseOrdinal: CaseOrdinal, durationSeconds: number,
    private readonly scheduler: FrameScheduler = browserScheduler) {
    this.clock = new ReplayClock(durationSeconds, scheduler.now);
    this.state = { caseOrdinal, durationSeconds, currentSeconds: 0, isPlaying: false,
      playbackSpeed: 30, replayMode: 'live', visibleWindow: visibleWindow(0, durationSeconds), ready: false };
  }
  getSnapshot = () => this.state;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener); return () => { this.listeners.delete(listener); };
  };
  subscribeFrame = (listener: (current: number) => void) => {
    this.frames.add(listener); return () => { this.frames.delete(listener); };
  };
  private publish(update: Partial<ReplayState> = {}) {
    const currentSeconds = this.clock.position();
    this.state = { ...this.state, ...update, currentSeconds,
      visibleWindow: this.chartDomain ?? visibleWindow(currentSeconds, this.state.durationSeconds) };
    this.lastPublished = this.scheduler.now();
    this.diagnostics.publications++;
    this.listeners.forEach(listener => listener());
    this.frames.forEach(listener => listener(currentSeconds));
  }
  beginChartScrub(domain: readonly [number, number]): boolean {
    if (!this.state.ready || this.chartDomain || !domain.every(Number.isFinite) || domain[1] <= domain[0]) return false;
    const bounded = [Math.max(0, domain[0]), Math.min(this.state.durationSeconds, domain[1])] as const;
    if (bounded[1] <= bounded[0]) return false;
    this.pause(); this.chartDomain = bounded; this.publish(); return true;
  }
  endChartScrub() { if (this.chartDomain) { this.chartDomain = undefined; this.publish(); } }
  activate() { if (!this.state.ready) this.publish({ ready: true }); }
  play() {
    if (this.hidden || !this.state.ready || this.state.isPlaying || this.clock.position() >= this.state.durationSeconds) return;
    this.endChartScrub(); this.clock.play(); this.publish({ isPlaying: true });
    this.handle = this.scheduler.request(this.tick);
  }
  pause() {
    if (!this.state.isPlaying) return;
    this.clock.pause(); this.cancel(); this.publish({ isPlaying: false });
  }
  visibilityChanged(hidden: boolean) { this.hidden = hidden; if (hidden) { this.pause(); this.endChartScrub(); } }
  toggle() { if (this.state.isPlaying) this.pause(); else this.play(); }
  seek(seconds: number) {
    if (!this.state.ready) return;
    this.clock.seek(seconds); this.cancel(); this.publish({ isPlaying: false });
  }
  setSpeed(speed: PlaybackSpeed) {
    if (!playbackSpeeds.includes(speed)) return;
    this.clock.setSpeed(speed); this.publish({ playbackSpeed: speed });
  }
  setMode(replayMode: ReplayMode) { if (replayMode !== this.state.replayMode) this.publish({ replayMode }); }
  private cancel() {
    if (this.handle !== undefined) this.scheduler.cancel(this.handle);
    this.handle = undefined;
  }
  private tick = () => {
    this.handle = undefined;
    if (!this.state.isPlaying) return;
    const current = this.clock.position();
    this.diagnostics.frames++;
    if (current >= this.state.durationSeconds) {
      this.clock.pause(); this.publish({ isPlaying: false }); return;
    }
    // Cursor subscribers are imperative; React and the data slices publish at most 10 Hz.
    this.frames.forEach(listener => listener(current));
    if (this.scheduler.now() - this.lastPublished >= PUBLICATION_INTERVAL_MS) this.publish();
    this.handle = this.scheduler.request(this.tick);
  };
  dispose() { this.pause(); this.endChartScrub(); this.cancel(); }
}
export function useReplay(store: ReplayStore) {
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
}
