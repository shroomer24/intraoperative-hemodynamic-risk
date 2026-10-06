export interface ScrubBounds { left: number; width: number; domain: readonly [number, number] }
// Pure display-coordinate mapping. It never rounds or reads observations/predictions.
export function pointerTime(clientX: number, bounds: ScrubBounds, duration: number): number {
  const fraction = bounds.width > 0 ? Math.min(1, Math.max(0, (clientX - bounds.left) / bounds.width)) : 0;
  return Math.min(duration, Math.max(0, bounds.domain[0] + fraction * (bounds.domain[1] - bounds.domain[0])));
}
export function elapsedFraction(current: number, duration: number): number { return duration > 0 ? current / duration : 0; }
