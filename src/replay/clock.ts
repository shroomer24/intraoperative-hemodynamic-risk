// Presentation time only. Every position derives from one monotonic wall-clock anchor.
export const playbackSpeeds = [1, 10, 30, 60] as const;
export type PlaybackSpeed = typeof playbackSpeeds[number];
export interface FrameScheduler {
  now(): number;
  request(callback: FrameRequestCallback): number;
  cancel(handle: number): void;
}
export const browserScheduler: FrameScheduler = {
  now: () => performance.now(),
  request: callback => requestAnimationFrame(callback),
  cancel: handle => cancelAnimationFrame(handle),
};
export class ReplayClock {
  private base = 0;
  private anchor = 0;
  private playing = false;
  private speed: PlaybackSpeed = 30;
  constructor(readonly duration: number, private readonly now: () => number) {}
  position(): number {
    return Math.min(this.duration, this.base + (this.playing
      ? Math.max(0, this.now() - this.anchor) * this.speed / 1000 : 0));
  }
  play() { this.anchor = this.now(); this.playing = this.base < this.duration; }
  pause() { this.base = this.position(); this.playing = false; }
  seek(seconds: number) {
    this.playing = false;
    this.base = Math.min(this.duration, Math.max(0, Number.isFinite(seconds) ? seconds : 0));
  }
  setSpeed(speed: PlaybackSpeed) {
    this.base = this.position(); this.anchor = this.now(); this.speed = speed;
  }
}
