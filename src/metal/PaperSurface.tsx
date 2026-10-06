import { useEffect, useId, useRef, useState } from 'react';
import { ShaderMount, liquidMetalFragmentShader } from '@paper-design/shaders';
import { fillConfig, pixelBudget } from './argentConfig';
import { permitsPaper } from './capability';
import { surfaceBudget, useMetalEnvironment } from './lifecycle';
import type { SurfaceKind } from './lifecycle';
import { paperUniforms } from './argentConfig';
import { useReplayMetalMotion } from './ReplayMetalMotion';
import { StaticMetalFallback } from './StaticMetalFallback';

export function PaperSurface({ kind }: { kind: SurfaceKind }) {
  // A transient surface exists only while its owning pill is hovered/focused.
  const motion = useReplayMetalMotion() || kind === 'transient'; const shader = useRef<ShaderMount | undefined>(undefined);
  const host = useRef<HTMLSpanElement>(null); const environment = useMetalEnvironment(); const id = useId();
  const [visible, setVisible] = useState(false); const [failed, setFailed] = useState(false); const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    if (typeof IntersectionObserver === 'undefined') { setVisible(true); return; }
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: '0px' });
    observer.observe(element); return () => observer.disconnect();
  }, []);
  const permitted = permitsPaper(environment, visible) && !failed;
  useEffect(() => {
    const element = host.current;
    if (!permitted || !element) return;
    let mount: ShaderMount | undefined; let resize: ResizeObserver | undefined; let stopped = false;
    const dispose = () => {
      if (stopped) return; stopped = true; resize?.disconnect();
      const canvas = mount?.canvasElement ?? element.querySelector('canvas');
      const gl = canvas?.getContext('webgl2');
      mount?.dispose(); if (shader.current === mount) shader.current = undefined; canvas?.remove();
      // Paper dispose cleans resources/observers but does not explicitly release the WebGL context.
      if (gl && !gl.isContextLost()) gl.getExtension('WEBGL_lose_context')?.loseContext(); setMounted(false);
    };
    const release = surfaceBudget.acquire(kind, id, dispose);
    const lost = (event: Event) => { event.preventDefault(); setFailed(true); dispose(); release(); };
    element.addEventListener('webglcontextlost', lost, true);
    try {
      const rect = element.getBoundingClientRect();
      mount = new ShaderMount(element, liquidMetalFragmentShader, paperUniforms(),
        { alpha: true, antialias: false, preserveDrawingBuffer: false, powerPreference: 'low-power' },
        fillConfig.speed, 0, 1, pixelBudget(rect.width, rect.height));
      shader.current = mount; mount.setSpeed?.(motion ? fillConfig.speed : 0);
      mount.canvasElement.dataset.metalCanvas = kind;
      // Area-derived maxPixelCount caps actual render density, including Retina/zoom.
      resize = new ResizeObserver(() => {
        if (stopped) return;
        const size = element.getBoundingClientRect(); mount?.setMaxPixelCount(pixelBudget(size.width, size.height));
      });
      resize.observe(element); setMounted(true);
    } catch { setFailed(true); dispose(); release(); }
    return () => { element.removeEventListener('webglcontextlost', lost, true); dispose(); release(); };
  }, [permitted, id, kind]);
  useEffect(() => { shader.current?.setSpeed?.(motion ? fillConfig.speed : 0); }, [motion]);
  return <span ref={host} className="paper-surface" aria-hidden="true" data-engine="paper" data-tone="gunmetal"
    data-surface-kind={kind} data-renderer={mounted && permitted ? 'paper' : 'static'}
    data-metal-reduced={environment.reduced || undefined}
    data-metal-motion={mounted && permitted && motion ? 'animated' : 'static'} data-paper-speed={mounted && permitted && motion ? fillConfig.speed : 0} data-context-failed={failed || undefined}>
    <StaticMetalFallback />
  </span>;
}
