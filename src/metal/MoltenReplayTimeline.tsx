import { useLayoutEffect, useRef } from 'react';
import { formatDuration } from '../data/types';
import type { ReplayState, ReplayStore } from '../replay/store';
import { elapsedFraction } from '../replay/scrub';
import { usePointerScrub } from '../replay/usePointerScrub';
import { PaperSurface } from './PaperSurface';

export function MoltenReplayTimeline({ store, state }: { store: ReplayStore; state: ReplayState }) {
  const input = useRef<HTMLInputElement>(null);
  const fill = useRef<HTMLSpanElement>(null); const thumb = useRef<HTMLSpanElement>(null); const host = useRef<HTMLDivElement>(null);
  const fraction = elapsedFraction(state.currentSeconds, state.durationSeconds);
  useLayoutEffect(() => store.subscribeFrame(seconds => {
    const next = elapsedFraction(seconds, state.durationSeconds);
    if (fill.current) fill.current.style.clipPath = `inset(0 ${100 - next * 100}% 0 0)`;
    if (thumb.current) thumb.current.style.left = `calc(8px + (100% - 16px) * ${next})`;
    if (host.current) host.current.dataset.elapsedFraction = String(next);
    if (input.current) { input.current.value = String(seconds); input.current.setAttribute('aria-valuetext', `${formatDuration(seconds)} of ${formatDuration(state.durationSeconds)}`); }
  }), [store, state.durationSeconds]);
  const scrub = usePointerScrub(store, element => {
    const rect = element.getBoundingClientRect(); return { left: rect.left + 8, width: rect.width - 16, domain: [0, state.durationSeconds] };
  }, false);
  return <div className="molten-time" ref={host} data-tone="gunmetal" data-elapsed-fraction={fraction} data-duration-seconds={state.durationSeconds}>
    <span className="molten-time-track" aria-hidden="true"><span ref={fill} className="molten-time-fill" style={{clipPath:`inset(0 ${100 - fraction * 100}% 0 0)`}}><PaperSurface kind="timeline" /></span></span>
    <span ref={thumb} className="molten-time-thumb" aria-hidden="true" style={{left:`calc(8px + (100% - 16px) * ${fraction})`}} />
    <input ref={input} id="surgical-timeline" type="range" min={0} max={state.durationSeconds} step="any" value={state.currentSeconds} disabled={!state.ready}
      aria-valuetext={`${formatDuration(state.currentSeconds)} of ${formatDuration(state.durationSeconds)}`} aria-describedby="replay-shortcuts" {...scrub}
      onChange={event => store.seek(Number(event.currentTarget.value))}
      onKeyDown={event => {
        const current = store.getSnapshot(); const delta = event.shiftKey ? 10 : 1;
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? current.durationSeconds
          : ['ArrowRight','ArrowUp'].includes(event.key) ? current.currentSeconds + delta
          : ['ArrowLeft','ArrowDown'].includes(event.key) ? current.currentSeconds - delta : undefined;
        if (next !== undefined) { event.preventDefault(); store.endChartScrub(); store.seek(next); }
      }} />
  </div>;
}
