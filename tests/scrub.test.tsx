import { existsSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach,beforeEach,describe,expect,it,vi } from 'vitest';
import { pointerTime, elapsedFraction } from '../src/replay/scrub';
import { ReplayStore } from '../src/replay/store';
import { PlaybackControls } from '../src/components/PlaybackControls';
import { SignalPanel } from '../src/components/SignalPanel';
import { RiskPanel } from '../src/components/RiskPanel';
import { SurfaceBudget } from '../src/metal/lifecycle';
import { ManualScheduler } from './replay.fixture';
import { outcomeFixture } from './outcomes.fixture';
import { signalFixture } from './signals.fixture';
import { metadata } from './metadata.fixture';
const plots=vi.hoisted(()=>[] as {data:unknown;setData:ReturnType<typeof vi.fn>;destroy:ReturnType<typeof vi.fn>}[]);
vi.mock('uplot',()=>({default:class {
  static pxRatio=1;static paths={stepped:()=>vi.fn()};root=document.createElement('div');bbox={left:36,top:8,width:600,height:58};
  setSize=vi.fn();setData=vi.fn();setScale=vi.fn();redraw=vi.fn();destroy=vi.fn(()=>this.root.remove());
  constructor(_options:unknown,public data:unknown,target:HTMLElement){target.append(this.root);plots.push(this);}
}}));
class ResizeMock { constructor(public callback:ResizeObserverCallback){} observe(target:Element){this.callback([{target,contentRect:{width:720}} as ResizeObserverEntry],this as unknown as ResizeObserver);}disconnect(){} }
for(const name of ['setPointerCapture','releasePointerCapture','hasPointerCapture'])if(!(name in HTMLElement.prototype))Object.defineProperty(HTMLElement.prototype,name,{configurable:true,writable:true,value:()=>false});
beforeEach(()=>{plots.length=0;vi.stubGlobal('ResizeObserver',ResizeMock);vi.spyOn(HTMLElement.prototype,'setPointerCapture').mockImplementation(()=>{});vi.spyOn(HTMLElement.prototype,'releasePointerCapture').mockImplementation(()=>{});vi.spyOn(HTMLElement.prototype,'hasPointerCapture').mockReturnValue(true);});
afterEach(()=>vi.unstubAllGlobals());
// jsdom does not implement PointerEvent; retain real coordinate/button/id semantics on synthetic events.
function pointer(element:Element,type:string,x:number,y=20,extras:Record<string,unknown>={}){
  const event=new MouseEvent(type,{bubbles:true,cancelable:true,clientX:x,clientY:y,button:0});Object.assign(event,{pointerId:1,pointerType:'mouse',isPrimary:true,...extras});fireEvent(element,event);
}
function rect(element:Element,left=100,width=600){vi.spyOn(element,'getBoundingClientRect').mockReturnValue({left,width,top:0,height:58,right:left+width,bottom:58,x:left,y:0,toJSON(){return{};}});}
async function physiology(){const fixture=signalFixture();const scheduler=new ManualScheduler();const store=new ReplayStore('case-004',7500,scheduler);const result=render(<SignalPanel selectedCase={fixture.selectedCase} loader={async()=>fixture.signals} store={store}/>);await screen.findByText('Verified monitor traces');return {...result,store,scheduler,fixture,hits:[...result.container.querySelectorAll('.chart-scrub-hit')]};}
function timeline(){const scheduler=new ManualScheduler();const store=new ReplayStore('case-001',15900,scheduler);store.activate();function Harness(){return <PlaybackControls store={store}/>;}const result=render(<Harness/>);const input=screen.getByRole('slider') as HTMLInputElement;rect(input,100,616);return {...result,store,scheduler,input};}
describe('exact chronological coordinate mapping',()=>{
  it('maps frozen plot-domain coordinates without rounding and clamps both edges',()=>{
    const bounds={left:100,width:600,domain:[1500,2400] as const};expect(pointerTime(100,bounds,7500)).toBe(1500);expect(pointerTime(700,bounds,7500)).toBe(2400);expect(pointerTime(123.4,bounds,7500)).toBeCloseTo(1535.1,12);expect(pointerTime(-100,bounds,7500)).toBe(1500);expect(pointerTime(9999,bounds,7500)).toBe(2400);expect(pointerTime(700,{...bounds,domain:[0,900]},300)).toBe(300);
  });
  it('uses chronological current/duration, not model probability, with exact start/end',()=>{expect(elapsedFraction(0,2100)).toBe(0);expect(elapsedFraction(2100,2100)).toBe(1);expect(elapsedFraction(1200,2100)).toBe(1200/2100);expect(elapsedFraction(3183,15900)).toBe(3183/15900);expect(elapsedFraction(0,0)).toBe(0);});
  it('preserves the original normal rolling domain and rejects conflicting/invalid freezes',()=>{
    const scheduler=new ManualScheduler();const store=new ReplayStore('case-001',15900,scheduler);store.activate();store.seek(2400);expect(store.beginChartScrub([1500,2400])).toBe(true);expect(store.beginChartScrub([1500,2400])).toBe(false);store.seek(1650.125);expect(store.getSnapshot().visibleWindow).toEqual([1500,2400]);store.endChartScrub();expect(store.getSnapshot().visibleWindow).toEqual([750.125,1650.125]);expect(store.beginChartScrub([NaN,900])).toBe(false);expect(store.beginChartScrub([900,0])).toBe(false);
  });
});
describe('Molten whole-operation semantic scrubber',()=>{
  it('retains semantic range min/max/value and direct exact elapsed fill',()=>{
    const {input,store,container}=timeline();expect(input.min).toBe('0');expect(input.max).toBe('15900');expect(input.step).toBe('any');act(()=>store.seek(3183.125));expect(input.value).toBe('3183.125');expect(container.querySelector('.molten-time')?.getAttribute('data-elapsed-fraction')).toBe(String(3183.125/15900));
    expect(container.querySelector<HTMLElement>('.molten-time-fill')?.style.clipPath).toBe(`inset(0 ${100-3183.125/15900*100}% 0 0)`);act(()=>store.seek(15900));expect(container.querySelector<HTMLElement>('.molten-time-fill')?.style.clipPath).toBe('inset(0 0% 0 0)');
  });
  it('click/drag seek the existing clock, pause, and remain paused on release',()=>{
    const {input,store,scheduler}=timeline();act(()=>store.play());pointer(input,'pointerdown',408);expect(store.getSnapshot()).toMatchObject({currentSeconds:7950,isPlaying:false});expect(scheduler.pending.size).toBe(0);pointer(input,'pointermove',258.125);expect(store.getSnapshot().currentSeconds).toBeCloseTo(3978.3125,10);pointer(input,'pointerup',258.125);expect(store.getSnapshot().isPlaying).toBe(false);expect(input.dataset.scrubbing).toBeUndefined();
  });
  it('seeks full-operation start/end and exposes exact keyboard state',()=>{
    const {input,store}=timeline();pointer(input,'pointerdown',-100);pointer(input,'pointerup',-100);expect(store.getSnapshot().currentSeconds).toBe(0);pointer(input,'pointerdown',9999);pointer(input,'pointerup',9999);expect(store.getSnapshot().currentSeconds).toBe(15900);fireEvent.keyDown(input,{key:'Home'});fireEvent.keyDown(input,{key:'ArrowRight',shiftKey:true});expect(store.getSnapshot().currentSeconds).toBe(10);expect(input.getAttribute('aria-valuetext')).toBe('00:00:10 of 04:25:00');fireEvent.keyDown(input,{key:'End'});expect(input.value).toBe('15900');
  });
  it('subscribes to the existing frame clock with no independent RAF/model calculation',()=>{
    const {input,store,scheduler,container}=timeline();act(()=>{store.setSpeed(1);store.play();});act(()=>scheduler.frame(16));expect(input.value).toBe('0.016');expect(container.querySelector('.molten-time')?.getAttribute('data-elapsed-fraction')).toBe(String(.016/15900));expect(readFileSync('src/metal/MoltenReplayTimeline.tsx','utf8')).not.toMatch(/requestAnimationFrame|setInterval|forecastValue|calibrat|predict_proba/);
  });
  it('permits three persistent plus one transient context, disposing replacement owners first',()=>{
    const budget=new SurfaceBudget();const stop=vi.fn();budget.acquire('mercury','m',stop);budget.acquire('probability','p',stop);budget.acquire('timeline','l',stop);budget.acquire('transient','t',stop);expect(budget.counts()).toEqual({persistent:3,transient:1,total:4});budget.acquire('transient','n',stop);expect(stop).toHaveBeenCalledOnce();expect(budget.counts().total).toBe(4);
  });
});
describe('synchronized physiology direct manipulation',()=>{
  it('clicks any plotting point and moves all four cursor/readouts without interpolation',async()=>{
    const {hits,store,container}=await physiology();hits.forEach(hit=>rect(hit));pointer(hits[0],'pointerdown',154.6);pointer(hits[0],'pointerup',154.6);expect(store.getSnapshot().currentSeconds).toBeCloseTo(81.9,12);expect(screen.getByLabelText('MAP at 00:01:21: unavailable').textContent).toBe('—');expect([...container.querySelectorAll<HTMLElement>('.signal-chart')].map(p=>Number(p.dataset.cursorSeconds))).toEqual(Array(4).fill(store.getSnapshot().currentSeconds));expect([...container.querySelectorAll<HTMLElement>('.signal-chart')].map(p=>p.dataset.renderedLast)).toEqual(Array(4).fill('81'));
  });
  it('freezes the shared domain during drag and restores normal rolling only at release',async()=>{
    const {hits,store,container,scheduler}=await physiology();act(()=>{store.seek(2400);store.play();});rect(hits[0]);pointer(hits[0],'pointerdown',400);expect(store.getSnapshot()).toMatchObject({currentSeconds:1950,isPlaying:false,visibleWindow:[1500,2400]});expect(scheduler.pending.size).toBe(0);
    pointer(hits[0],'pointermove',200);expect(store.getSnapshot().currentSeconds).toBe(1650);expect([...container.querySelectorAll<HTMLElement>('.signal-chart')].map(p=>[p.dataset.domainStart,p.dataset.domainEnd])).toEqual(Array(4).fill(['1500','2400']));
    pointer(hits[0],'pointerup',200);expect(store.getSnapshot().visibleWindow).toEqual([750,1650]);expect(hits[0].getAttribute('data-scrub-mapped-seconds')).toBe('1650');expect(hits[0].getAttribute('data-scrub-result-seconds')).toBe('1650');expect(plots).toHaveLength(4);
  });
  it.each(['map','hr','spo2','etco2'])('allows %s drag to seek exactly the same shared clock',async channel=>{
    const {hits,store,container}=await physiology();const hit=hits[['map','hr','spo2','etco2'].indexOf(channel)];rect(hit);pointer(hit,'pointerdown',400);pointer(hit,'pointermove',500);pointer(hit,'pointerup',500);expect(store.getSnapshot().currentSeconds).toBe(600);expect([...container.querySelectorAll<HTMLElement>('.signal-chart')].every(p=>p.dataset.cursorSeconds==='600')).toBe(true);
  });
  it.each(['pointercancel','lostpointercapture'])('%s releases frozen domain without resuming or committing an extra coordinate',async event=>{
    const {hits,store}=await physiology();act(()=>store.seek(2400));rect(hits[0]);pointer(hits[0],'pointerdown',400);pointer(hits[0],event,600);expect(store.getSnapshot()).toMatchObject({currentSeconds:1950,isPlaying:false,visibleWindow:[1050,1950]});expect(hits[0].getAttribute('data-scrubbing')).toBeNull();
  });
  it('clamps at both frozen edges, keeps Live masked and Review unchanged',async()=>{
    const {hits,store,container}=await physiology();rect(hits[0]);pointer(hits[0],'pointerdown',-100);expect(store.getSnapshot().currentSeconds).toBe(0);pointer(hits[0],'pointermove',9999);expect(store.getSnapshot().currentSeconds).toBe(900);pointer(hits[0],'pointermove',400);expect([...container.querySelectorAll<HTMLElement>('.signal-chart')].every(p=>p.dataset.renderedLast==='450')).toBe(true);act(()=>store.setMode('review'));expect([...container.querySelectorAll<HTMLElement>('.signal-chart')].every(p=>p.dataset.renderedLast==='900')).toBe(true);pointer(hits[0],'pointerup',400);expect(store.getSnapshot().replayMode).toBe('review');
  });
  it('touch vertical intent safely cancels without blocking page scrolling',async()=>{
    const {hits,store}=await physiology();rect(hits[0]);pointer(hits[0],'pointerdown',400,20,{pointerType:'touch'});pointer(hits[0],'pointermove',401,80,{pointerType:'touch'});expect(hits[0].getAttribute('data-scrubbing')).toBeNull();expect(store.getSnapshot().visibleWindow).toEqual([0,900]);expect(readFileSync('src/styles/interaction.css','utf8')).toContain('touch-action: pan-y');
  });
  it('unmount safely terminates the session; no clock callback or plot/context leak remains',async()=>{
    const {hits,store,unmount,scheduler}=await physiology();act(()=>store.seek(2400));rect(hits[0]);pointer(hits[0],'pointerdown',400);unmount();expect(store.getSnapshot().visibleWindow).toEqual([1050,1950]);expect(plots.every(p=>p.destroy.mock.calls.length===1)).toBe(true);expect(scheduler.pending.size).toBe(0);
  });
});
describe('preserved scientific selection and causal outcomes while seeking',()=>{
  it('keeps exact predictions/probability geometry/event confirmation and Current MAP through a freeze',()=>{
    const f=outcomeFixture();const store=new ReplayStore('case-004',7500,new ManualScheduler());store.activate();store.seek(1576);
    const props={store,predictions:{status:'ready' as const,ordinal:f.selectedCase.ordinal,data:f.predictions},outcomes:{status:'ready' as const,ordinal:f.selectedCase.ordinal,data:f.outcomes},events:{status:'ready' as const,ordinal:f.selectedCase.ordinal,data:f.events},representation:'calibrated' as const,onRepresentationChange:()=>{}};
    const {container,rerender}=render(<RiskPanel {...props} model={metadata.models.find(m=>m.id==='tabpfn_full')!}/>);const probability=container.querySelector('.risk-panel')?.getAttribute('data-value');expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('pending');
    act(()=>{store.beginChartScrub([676,1576]);store.seek(1577);});expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('positive');expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).toBe(probability);expect(container.querySelector('.molten-progress')?.getAttribute('data-probability')).toBe(probability);
    rerender(<RiskPanel {...props} model={metadata.models.find(m=>m.id==='current_map')!}/>);expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).toBe('-80');expect(container.querySelector('.molten-progress')).toBeNull();act(()=>store.endChartScrub());
  });
  it('keeps all scientific selectors, package lock, original plot options and frozen sources unchanged',()=>{
    const hashes=JSON.parse(readFileSync('tests/fixtures/phase61-source-hashes.json','utf8'));
    // The private research exporter is deliberately absent from public checkouts.
    // Its original hash assertion still runs in the authoritative local workspace.
    for(const [name,digest] of Object.entries(hashes))if((name!=='tools/export_replay.py'||existsSync(name))&&(name.startsWith('src/data/')||name.startsWith('src/predictions/')||name.startsWith('src/outcomes/')||['src/replay/clock.ts','src/replay/viewport.ts','src/replay/visibility.ts','src/charts/signalOptions.ts','src/charts/RiskHistoryPlot.tsx','src/charts/riskOptions.ts','src/charts/synchronizedAxes.ts','src/components/ForecastOutcome.tsx','src/components/EventLane.tsx','src/metal/MoltenProbabilityProgress.tsx','package-lock.json','package.json','tools/export_replay.py'].includes(name)))expect(createHash('sha256').update(readFileSync(name)).digest('hex')).toBe(digest);
  });
});
