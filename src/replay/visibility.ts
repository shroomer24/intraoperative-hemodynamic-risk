import type { ReplayStore } from './store';

export function pauseWhenHidden(store: ReplayStore, doc: Document = document): () => void {
  const onVisibility = () => { store.visibilityChanged(doc.hidden); };
  doc.addEventListener('visibilitychange', onVisibility);
  onVisibility();
  return () => doc.removeEventListener('visibilitychange', onVisibility);
}
export function editableTarget(target: EventTarget | null): boolean {
  return target instanceof Element && !!target.closest('input, select, textarea, button, [contenteditable]:not([contenteditable="false"])');
}
export function replayKeyboard(store: ReplayStore, doc: Document = document): () => void {
  const onKey = (event: KeyboardEvent) => {
    if (editableTarget(event.target) || event.altKey || event.ctrlKey || event.metaKey || event.repeat && event.key === ' ') return;
    const state = store.getSnapshot();
    if (!state.ready) return;
    switch (event.key) {
      case ' ': store.toggle(); break;
      case 'Home': store.seek(0); break;
      case 'End': store.seek(state.durationSeconds); break;
      case 'ArrowLeft': store.seek(state.currentSeconds - (event.shiftKey ? 10 : 1)); break;
      case 'ArrowRight': store.seek(state.currentSeconds + (event.shiftKey ? 10 : 1)); break;
      default: return;
    }
    event.preventDefault();
  };
  doc.addEventListener('keydown', onKey);
  return () => doc.removeEventListener('keydown', onKey);
}
