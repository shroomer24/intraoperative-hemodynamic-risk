import type { FrameScheduler } from '../src/replay/clock';
export class ManualScheduler implements FrameScheduler {
  wall = 0;
  sequence = 0;
  pending = new Map<number, FrameRequestCallback>();
  now = () => this.wall;
  request = (callback: FrameRequestCallback) => { const id = ++this.sequence; this.pending.set(id, callback); return id; };
  cancel = (id: number) => { this.pending.delete(id); };
  frame(milliseconds: number) {
    this.wall += milliseconds;
    const callbacks = [...this.pending.values()]; this.pending.clear();
    callbacks.forEach(callback => callback(this.wall));
  }
}
