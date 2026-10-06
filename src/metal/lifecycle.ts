import { useEffect, useState, useSyncExternalStore } from 'react';
import { metalEnvironment } from './capability';
import type { MetalEnvironment } from './capability';

export function useMetalEnvironment(): MetalEnvironment {
  const [environment, setEnvironment] = useState(metalEnvironment);
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setEnvironment(metalEnvironment());
    document.addEventListener('visibilitychange', update); media.addEventListener('change', update); update();
    return () => { document.removeEventListener('visibilitychange', update); media.removeEventListener('change', update); };
  }, []);
  return environment;
}
let interactionOwner: string | undefined;
const listeners = new Set<() => void>();
export function claimInteraction(id: string) { interactionOwner = id; listeners.forEach(listener => listener()); }
export function releaseInteraction(id: string) {
  if (interactionOwner === id) { interactionOwner = undefined; listeners.forEach(listener => listener()); }
}
export function useInteractionOwner() {
  return useSyncExternalStore(listener => { listeners.add(listener); return () => listeners.delete(listener); }, () => interactionOwner);
}
export type SurfaceKind = 'mercury' | 'probability' | 'timeline' | 'transient';
export class SurfaceBudget {
  private active = new Map<SurfaceKind, { id: string; stop: () => void }>();
  acquire(kind: SurfaceKind, id: string, stop: () => void): () => void {
    // Synchronously dispose the prior owner BEFORE a replacement can create a context.
    this.active.get(kind)?.stop(); this.active.set(kind, { id, stop });
    return () => { if (this.active.get(kind)?.id === id) this.active.delete(kind); };
  }
  counts() { return { persistent: Number(this.active.has('mercury')) + Number(this.active.has('probability')) + Number(this.active.has('timeline')),
    transient: Number(this.active.has('transient')), total: this.active.size }; }
}
export const surfaceBudget = new SurfaceBudget();
