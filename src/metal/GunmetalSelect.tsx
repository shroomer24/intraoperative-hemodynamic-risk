import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties, KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import { InteractiveMetalTab } from './InteractiveMetalTab';

export interface GunmetalOption { value: string; label: string; disabled?: boolean; group?: string }
// Select-only combobox: DOM focus stays on the trigger, never on decorative menu rows.
export function GunmetalSelect({ id, label, value, options, onChange, disabled = false }: {
  id?: string; label: string; value: string; options: readonly GunmetalOption[];
  onChange: (value: string) => void; disabled?: boolean;
}) {
  const generated = useId(); const controlId = id ?? `gunmetal-${generated}`; const popupId = `${controlId}-listbox`;
  const trigger = useRef<HTMLButtonElement>(null); const popup = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false); const [active, setActive] = useState(0);
  const [placement, setPlacement] = useState<CSSProperties>({}); const typing = useRef({ text: '', time: 0 });
  const enabled = options.map((option, index) => option.disabled ? -1 : index).filter(index => index >= 0);
  const selected = options.findIndex(option => option.value === value);
  const optionId = (index: number) => `${popupId}-option-${index}`;
  const show = (index = selected) => {
    if (disabled || !enabled.length) return;
    setActive(enabled.includes(index) ? index : enabled[0]); setOpen(true);
  };
  const choose = (index: number) => {
    if (!options[index] || options[index].disabled) return;
    onChange(options[index].value); setOpen(false); trigger.current?.focus({ preventScroll: true });
  };
  useLayoutEffect(() => {
    if (!open) return;
    const position = () => {
      const rect = trigger.current!.getBoundingClientRect(); const margin = 12; const gap = 6;
      const below = window.innerHeight - rect.bottom - margin; const above = rect.top - margin;
      const up = below < 180 && above > below; const room = Math.max(0, up ? above : below);
      const width = Math.min(rect.width, window.innerWidth - margin * 2);
      setPlacement({ position: 'fixed', left: Math.max(margin, Math.min(rect.left, window.innerWidth - width - margin)), width,
        maxHeight: Math.min(320, Math.max(0, room - gap)), ...(up ? { bottom: window.innerHeight - rect.top + gap } : { top: rect.bottom + gap }) });
    };
    const scroll = (event: Event) => { if (!popup.current?.contains(event.target as Node)) position(); };
    position(); window.addEventListener('resize', position); window.addEventListener('scroll', scroll, true);
    return () => { window.removeEventListener('resize', position); window.removeEventListener('scroll', scroll, true); };
  }, [open]);
  useLayoutEffect(() => {
    if (!open) return;
    const row = document.getElementById(optionId(active)); const menu = popup.current;
    if (!row || !menu) return;
    // Internal scrolling only; never scroll the operation page to the active option.
    if (row.offsetTop < menu.scrollTop) menu.scrollTop = row.offsetTop;
    else if (row.offsetTop + row.offsetHeight > menu.scrollTop + menu.clientHeight) menu.scrollTop = row.offsetTop + row.offsetHeight - menu.clientHeight;
  }, [open, active, placement]);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      const node = event.target as Node;
      if (!trigger.current?.contains(node) && !popup.current?.contains(node)) setOpen(false);
    };
    document.addEventListener('pointerdown', outside, true);
    return () => document.removeEventListener('pointerdown', outside, true);
  }, [open]);
  useEffect(() => { if (disabled) setOpen(false); }, [disabled]);
  const key = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === 'Tab') { setOpen(false); return; }
    if (event.key === 'Escape') { event.preventDefault(); setOpen(false); return; }
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); if (open) choose(active); else show(); return; }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      if (!enabled.length) return;
      if (event.key === 'Home') show(enabled[0]);
      else if (event.key === 'End') show(enabled.at(-1));
      else if (!open) show();
      else { const index = enabled.indexOf(active); setActive(enabled[Math.min(enabled.length - 1, Math.max(0, index + (event.key === 'ArrowDown' ? 1 : -1)))]); }
      return;
    }
    if (event.key.length === 1 && /\S/.test(event.key)) {
      event.preventDefault(); const now = performance.now(); const previous = now - typing.current.time < 700 ? typing.current.text : '';
      const text = (previous + event.key).toLocaleLowerCase(); typing.current = { text, time: now };
      const prefix = [...text].every(char => char === text[0]) ? text[0] : text;
      const start = open ? active : selected;
      const candidates = [...enabled.filter(index => index > start), ...enabled.filter(index => index <= start)];
      const match = candidates.find(index => options[index].label.toLocaleLowerCase().startsWith(prefix));
      if (match !== undefined) show(match);
    }
  };
  const row = (option: GunmetalOption, index: number) => <div key={option.value} id={optionId(index)} role="option"
    aria-selected={option.value === value} aria-disabled={option.disabled || undefined} data-value={option.value}
    data-active={index === active || undefined} className="gunmetal-option"
    onPointerMove={() => { if (!option.disabled) setActive(index); }} onMouseDown={event => event.preventDefault()}
    onClick={() => choose(index)}><span>{option.label}</span><span className="gunmetal-selected" aria-hidden="true">{option.value === value ? '✓' : ''}</span></div>;
  const groups: { group?: string; items: { option: GunmetalOption; index: number }[] }[] = [];
  options.forEach((option, index) => {
    if (!groups.length || groups.at(-1)!.group !== option.group) groups.push({ group: option.group, items: [] });
    groups.at(-1)!.items.push({ option, index });
  });
  return <>
    <InteractiveMetalTab select active={open}><button ref={trigger} id={controlId} type="button" role="combobox" aria-label={label}
      value={value} data-value={value} aria-expanded={open} aria-haspopup="listbox" aria-controls={popupId}
      aria-activedescendant={open ? optionId(active) : undefined} disabled={disabled}
      className="gunmetal-select-trigger" onClick={() => open ? setOpen(false) : show()} onKeyDown={key}
      onBlur={event => { if (!popup.current?.contains(event.relatedTarget)) setOpen(false); }}>
      <span>{options[selected]?.label ?? 'Unavailable'}</span>
    </button></InteractiveMetalTab>
    {open && createPortal(<div ref={popup} id={popupId} className="gunmetal-listbox" role="listbox" aria-label={label} style={placement} data-tone="gunmetal">
      {groups.map((group, index) => group.group ? <div key={index} role="group" aria-label={group.group}>
        <div className="gunmetal-group" aria-hidden="true">{group.group}</div>{group.items.map(({option,index}) => row(option,index))}
      </div> : group.items.map(({option,index}) => row(option,index)))}
    </div>, document.body)}
  </>;
}
