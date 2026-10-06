import { Children, isValidElement, useEffect, useId, useState } from 'react';
import type { ReactElement, ReactNode } from 'react';
import { claimInteraction, releaseInteraction, useInteractionOwner } from './lifecycle';
import { PaperSurface } from './PaperSurface';
import { StaticMetalFallback } from './StaticMetalFallback';

export function InteractiveMetalTab({ children, select = false, active = false }: { children: ReactNode; select?: boolean; active?: boolean }) {
  const child = Children.only(children) as ReactElement<{ disabled?: boolean }>;
  const disabled = isValidElement(child) && Boolean(child.props.disabled); const id = useId();
  const owner = useInteractionOwner(); const [hover, setHover] = useState(false); const [focus, setFocus] = useState(false);
  const engaged = !disabled && (hover || focus); const enhanced = engaged && owner === id;
  useEffect(() => { if (!engaged) releaseInteraction(id); }, [engaged, id]);
  useEffect(() => () => releaseInteraction(id), [id]);
  return <span className={`metal-tab${select ? ' metal-select' : ' metal-button'}`} data-tone="gunmetal" data-engine="paper"
    data-variant="border" data-frame="single" data-finish="surface" data-active={active || undefined}
    data-engaged={engaged || undefined} data-disabled={disabled || undefined} data-transient={enhanced || undefined}
    onPointerEnter={() => { if (!disabled) { setHover(true); claimInteraction(id); } }}
    onPointerLeave={() => { setHover(false); if (!focus) releaseInteraction(id); }}
    onFocusCapture={() => { if (!disabled) { setFocus(true); claimInteraction(id); } }}
    onBlurCapture={event => { if (!event.currentTarget.contains(event.relatedTarget)) { setFocus(false); if (!hover) releaseInteraction(id); } }}>
    <span className="metal-tab-decoration" aria-hidden="true"><StaticMetalFallback />
      {enhanced && <PaperSurface kind="transient" />}<span className="metal-tab-core" /><span className="metal-tab-sheen" />
    </span>
    {child}
    {select && <svg className="metal-select-chevron" width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="m4 6 4 4 4-4" stroke="currentColor" strokeWidth="1.3" /></svg>}
  </span>;
}
