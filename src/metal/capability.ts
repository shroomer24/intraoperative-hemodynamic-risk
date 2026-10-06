export interface MetalEnvironment { hidden: boolean; reduced: boolean; supported: boolean; forcedStatic: boolean }
export function metalEnvironment(): MetalEnvironment {
  const flags = new URLSearchParams(window.location.search);
  return { hidden: document.hidden, reduced: window.matchMedia('(prefers-reduced-motion: reduce)').matches || flags.get('metal') === 'still',
    supported: typeof window.WebGL2RenderingContext !== 'undefined' && typeof ResizeObserver !== 'undefined',
    forcedStatic: flags.get('metal') === 'static' };
}
export function permitsPaper(environment: MetalEnvironment, visible: boolean): boolean {
  return visible && environment.supported && !environment.forcedStatic && !environment.hidden && !environment.reduced;
}
