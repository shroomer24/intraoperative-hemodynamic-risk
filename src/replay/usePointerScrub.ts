import { useEffect, useRef } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import type { ReplayStore } from './store';
import { pointerTime } from './scrub';
import type { ScrubBounds } from './scrub';

export function usePointerScrub(store: ReplayStore | undefined, getBounds: (element: HTMLElement) => ScrubBounds, freezeDomain: boolean) {
  const session = useRef<{ id: number; bounds: ScrubBounds; element: HTMLElement; x: number; y: number; touch: boolean } | undefined>(undefined);
  const finish = () => {
    const drag = session.current; if (!drag) return;
    session.current = undefined; delete drag.element.dataset.scrubbing;
    if (drag.element.hasPointerCapture?.(drag.id)) drag.element.releasePointerCapture(drag.id);
    if (freezeDomain) store?.endChartScrub();
  };
  useEffect(() => {
    const hidden = () => { if (document.hidden) finish(); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape' && session.current) { event.preventDefault(); finish(); } };
    document.addEventListener('visibilitychange', hidden); document.addEventListener('keydown', escape); window.addEventListener('blur', finish);
    return () => { finish(); document.removeEventListener('visibilitychange', hidden); document.removeEventListener('keydown', escape); window.removeEventListener('blur', finish); };
  }, [store]);
  const seek = (event: ReactPointerEvent<HTMLElement>) => {
    const drag = session.current; if (!drag || event.pointerId !== drag.id || !store) return;
    const mapped = pointerTime(event.clientX, drag.bounds, store.getSnapshot().durationSeconds);
    store.seek(mapped);
    Object.assign(drag.element.dataset, { scrubPointerX: String(event.clientX), scrubMappedSeconds: String(mapped),
      scrubResultSeconds: String(store.getSnapshot().currentSeconds), scrubDomainStart: String(drag.bounds.domain[0]), scrubDomainEnd: String(drag.bounds.domain[1]) });
  };
  return {
    onPointerDown: (event: ReactPointerEvent<HTMLElement>) => {
      if (!store?.getSnapshot().ready || event.button !== 0 || event.isPrimary === false || session.current) return;
      const bounds = getBounds(event.currentTarget); if (bounds.width <= 0) return;
      if (freezeDomain) { if (!store.beginChartScrub(bounds.domain)) return; } else { store.endChartScrub(); store.pause(); }
      session.current = { id: event.pointerId, bounds: { ...bounds, domain: [...bounds.domain] }, element: event.currentTarget,
        x: event.clientX, y: event.clientY, touch: event.pointerType === 'touch' };
      event.currentTarget.setPointerCapture?.(event.pointerId); event.currentTarget.dataset.scrubbing = 'true';
      if (event.currentTarget instanceof HTMLInputElement) event.currentTarget.focus({ preventScroll: true });
      if (event.pointerType !== 'touch') event.preventDefault(); seek(event);
    },
    onPointerMove: (event: ReactPointerEvent<HTMLElement>) => {
      const drag = session.current; if (!drag || drag.id !== event.pointerId) return;
      if (drag.touch && Math.abs(event.clientY - drag.y) >= 4 && Math.abs(event.clientY - drag.y) > Math.abs(event.clientX - drag.x)) { finish(); return; }
      if (drag.touch && Math.abs(event.clientX - drag.x) < 4) return;
      if (event.cancelable) event.preventDefault(); seek(event);
    },
    onPointerUp: (event: ReactPointerEvent<HTMLElement>) => { if (session.current?.id === event.pointerId) { seek(event); finish(); } },
    onPointerCancel: (event: ReactPointerEvent<HTMLElement>) => { if (session.current?.id === event.pointerId) finish(); },
    onLostPointerCapture: (event: ReactPointerEvent<HTMLElement>) => { if (session.current?.id === event.pointerId) finish(); },
  };
}
