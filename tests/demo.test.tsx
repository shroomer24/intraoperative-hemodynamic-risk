import { StrictMode } from 'react';
import { act, render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { DemoStart } from '../src/demo/DemoStart';
import { demoPreset, isDemoPath, resolveDemoAnchor } from '../src/demo/preset';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
import { predictionFixture } from './predictions.fixture';
import type { PredictionLoadState } from '../src/data/casePredictions';
const data = predictionFixture('case-019').predictions;
const ready: PredictionLoadState = {status:'ready',ordinal:'case-019',data};
function harness() { const scheduler=new ManualScheduler();const store=new ReplayStore('case-019',2100,scheduler);store.activate();return {store,scheduler,pending:{current:true}}; }
describe('curated demo selection only',()=>{
  it.each(['/demo','/demo/'])('recognizes %s',path=>expect(isDemoPath(path)).toBe(true));
  it.each(['/','/demo-extra','/DEMO','/test','/demo/nested'])('preserves normal behavior for %s',path=>expect(isDemoPath(path)).toBe(false));
  it('resolves the approved retained forecast rather than displaying a synthetic value',()=>{
    expect(resolveDemoAnchor(data)).toBe(774);expect(demoPreset.modelId).toBe('tabpfn_full');
    const row=data.windows.find(w=>w.query_position===2765)!;expect(row.model_outputs.tabpfn_full?.calibrated_probability).toBe(0.3038048376358691);
  });
  it('rejects missing, duplicate, wrong-case or mismatched frozen identity',()=>{
    const row=data.windows.find(w=>w.query_position===2765)!;
    expect(resolveDemoAnchor({...data,ordinal:'case-001'})).toBeUndefined();
    expect(resolveDemoAnchor({...data,windows:[]})).toBeUndefined();
    expect(resolveDemoAnchor({...data,windows:[row,row]})).toBeUndefined();
    expect(resolveDemoAnchor({...data,windows:[{...row,model_outputs:{...row.model_outputs,tabpfn_full:null}}]})).toBeUndefined();
  });
  it('starts paused/live once, including StrictMode, and leaves subsequent seeking alone',()=>{
    const h=harness();const {rerender}=render(<StrictMode><DemoStart {...h} predictions={ready}/></StrictMode>);
    expect(h.store.getSnapshot()).toMatchObject({currentSeconds:774,isPlaying:false,replayMode:'live'});expect(h.scheduler.pending.size).toBe(0);
    act(()=>h.store.seek(900));rerender(<StrictMode><DemoStart {...h} predictions={{...ready}}/></StrictMode>);expect(h.store.getSnapshot().currentSeconds).toBe(900);h.store.dispose();
  });
  it.each(['seek','play','review'] as const)('does not override early manual %s while forecasts load',action=>{
    const h=harness();if(action==='seek')h.store.seek(600);if(action==='play')h.store.play();if(action==='review')h.store.setMode('review');
    render(<DemoStart {...h} predictions={ready}/>);expect(h.pending.current).toBe(false);expect(h.store.getSnapshot().currentSeconds).not.toBe(774);h.store.dispose();
  });
  it('fails safely on an unavailable frozen anchor',()=>{
    const h=harness();const {getByRole}=render(<DemoStart {...h} predictions={{...ready,data:{...data,windows:[]}}}/>);
    expect(getByRole('status').textContent).toContain('unavailable');expect(h.store.getSnapshot().currentSeconds).toBe(0);h.store.dispose();
  });
});
