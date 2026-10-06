import { chooseOption } from './select.fixture';
import { Profiler } from 'react';
import { webcrypto } from 'node:crypto';
import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useCaseEvents } from '../src/data/caseEvents';
import type { EventLoader, EventLoadState } from '../src/data/caseEvents';
import type { CaseEvents } from '../src/data/eventValidation';
import type { PredictionLoadState } from '../src/data/casePredictions';
import { forecastState, outcomeState, useCaseForecasts } from '../src/data/forecastOutcomes';
import type { ForecastBundle, ForecastLoader, OutcomeLoadState } from '../src/data/forecastOutcomes';
import type { PresentationCase } from '../src/data/types';
import { RiskPanel } from '../src/components/RiskPanel';
import { RiskHistory } from '../src/components/RiskHistory';
import { EventLane } from '../src/components/EventLane';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
import { outcomeFixture } from './outcomes.fixture';
import { metadata } from './metadata.fixture';
import App from '../src/App';

const mocks = vi.hoisted(()=>({plots:[] as {destroy:ReturnType<typeof vi.fn>}[]}));
vi.mock('uplot',()=>({default:class {
  static pxRatio=1;static paths={stepped:()=>()=>null};
  root=document.createElement('div');bbox={left:36,top:8,width:600,height:58};
  setData=vi.fn();setScale=vi.fn();setSize=vi.fn();redraw=vi.fn();destroy=vi.fn(()=>this.root.remove());
  constructor(_options:unknown,_data:unknown,target:HTMLElement){this.root.append(document.createElement('canvas'));target.append(this.root);mocks.plots.push(this);}
}}));
class ResizeMock {
  constructor(private callback:ResizeObserverCallback){}
  observe(){this.callback([{contentRect:{width:720}} as ResizeObserverEntry],this as unknown as ResizeObserver);}
  disconnect(){}
}
beforeEach(()=>{mocks.plots.length=0;vi.stubGlobal('ResizeObserver',ResizeMock);vi.stubGlobal('crypto',webcrypto);});
afterEach(()=>vi.unstubAllGlobals());
const fixture=outcomeFixture();
function states(value=fixture){
  return {predictions:{status:'ready',ordinal:value.selectedCase.ordinal,data:value.predictions} as PredictionLoadState,
    outcomes:{status:'ready',ordinal:value.selectedCase.ordinal,data:value.outcomes} as OutcomeLoadState,
    events:{status:'ready',ordinal:value.selectedCase.ordinal,data:value.events} as EventLoadState};
}
function harnessStore(value=fixture){const scheduler=new ManualScheduler();const store=new ReplayStore(value.selectedCase.ordinal,value.selectedCase.durationSeconds,scheduler);store.activate();return {store,scheduler};}
function deferred<T>(){let resolve!:(value:T)=>void;let reject!:(error:Error)=>void;const promise=new Promise<T>((ok,fail)=>{resolve=ok;reject=fail;});return {promise,resolve,reject};}
function LoaderHarness({selectedCase,loader,store,values}:{selectedCase:PresentationCase;loader:EventLoader;store:ReplayStore;values:ReturnType<typeof states>}){
  const events=useCaseEvents(selectedCase,loader);return <EventLane store={store} events={events} predictions={values.predictions} outcomes={values.outcomes}/>;
}
function BundleHarness({selectedCase,loader,store,events}:{selectedCase:PresentationCase;loader:ForecastLoader;store:ReplayStore;events:EventLoadState}){
  const state=useCaseForecasts(selectedCase,loader);
  return <RiskPanel model={metadata.models.find(model=>model.id==='tabpfn_full')!} store={store} predictions={forecastState(state)} outcomes={outcomeState(state)} events={events} representation="calibrated" onRepresentationChange={()=>{}}/>;
}
function panels(value=fixture,modelId='tabpfn_full'){
  const {store,scheduler}=harnessStore(value);const values=states(value);
  const result=render(<><RiskPanel model={metadata.models.find(model=>model.id===modelId)!} store={store} {...values} representation="calibrated" onRepresentationChange={()=>{}}/>
    <RiskHistory modelId="tabpfn_full" store={store} predictions={values.predictions} representation="calibrated"/>
    <EventLane store={store} {...values}/></>);
  return {...result,store,scheduler};
}

describe('historical event request identity',()=>{
  it('cancels combined forecasts/truth and rejects late old-case publication even when new signals/events are ready',async()=>{
    const old=deferred<ForecastBundle>();const next=deferred<ForecastBundle>();const nextFixture=outcomeFixture('case-019');
    const loader=vi.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(next.promise);
    const oldStore=harnessStore().store;oldStore.seek(1577);const nextStore=harnessStore(nextFixture).store;nextStore.seek(1200);
    const {container,rerender}=render(<BundleHarness selectedCase={fixture.selectedCase} loader={loader} store={oldStore} events={states().events}/>);
    const signal=loader.mock.calls[0][1] as AbortSignal;
    rerender(<BundleHarness selectedCase={nextFixture.selectedCase} loader={loader} store={nextStore} events={states(nextFixture).events}/>);
    expect(signal.aborted).toBe(true);
    await act(async()=>old.resolve({ordinal:'case-004',forecasts:fixture.predictions,outcomes:fixture.outcomes}));
    expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).toBeNull();
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-onset-seconds')).toBeNull();
    await act(async()=>next.resolve({ordinal:'case-019',forecasts:nextFixture.predictions,outcomes:nextFixture.outcomes}));
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-onset-seconds')).toBe('813');
    expect(container.querySelector('.risk-panel')?.getAttribute('data-query-position')).toBe('2765');
  });
  it('aborts a prior request and ignores its late success under the next operation',async()=>{
    const old=deferred<CaseEvents>();const next=deferred<CaseEvents>();const nextFixture=outcomeFixture('case-019');
    const loader=vi.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(next.promise);
    const firstStore=harnessStore().store;firstStore.seek(1600);const secondStore=harnessStore(nextFixture).store;secondStore.seek(1200);
    const {container,rerender}=render(<LoaderHarness selectedCase={fixture.selectedCase} loader={loader} store={firstStore} values={states()}/>);
    const signal=loader.mock.calls[0][1] as AbortSignal;
    rerender(<LoaderHarness selectedCase={nextFixture.selectedCase} loader={loader} store={secondStore} values={states(nextFixture)}/>);
    expect(signal.aborted).toBe(true);await act(async()=>old.resolve(fixture.events));
    expect(container.querySelectorAll('.event-band')).toHaveLength(0);expect(container.textContent).not.toContain('00:25:18');
    await act(async()=>next.resolve(nextFixture.events));
    expect(container.querySelector('.event-band')?.getAttribute('data-onset-seconds')).toBe('813');
    expect(container.querySelector('.event-lane')?.getAttribute('data-case-ordinal')).toBe('case-019');
  });
  it('immediately clears previously loaded events on case change and safely reports failure',async()=>{
    const nextFixture=outcomeFixture('case-019');const pending=deferred<CaseEvents>();
    const loader=vi.fn().mockResolvedValueOnce(fixture.events).mockReturnValueOnce(pending.promise);
    const store=harnessStore().store;store.seek(1600);
    const {container,rerender}=render(<LoaderHarness selectedCase={fixture.selectedCase} loader={loader} store={store} values={states()}/>);
    await screen.findByText('Event details · 1 confirmed');
    rerender(<LoaderHarness selectedCase={nextFixture.selectedCase} loader={loader} store={store} values={states(nextFixture)}/>);
    expect(container.querySelectorAll('.event-band')).toHaveLength(0);expect(container.textContent).not.toContain('00:25:18');
    await act(async()=>pending.reject(new Error('private detail')));
    expect(container.textContent).toContain('Historical events unavailable');expect(container.textContent).not.toContain('private detail');
  });
});

describe('causal event and selected forecast UI',()=>{
  it('publishes no future event/outcome fields at onset or before confirmation; at confirmation annotates exact onset',()=>{
    const {container,store}=panels();
    for(const time of [1517,1518,1576]){
      act(()=>store.seek(time));expect(container.querySelectorAll('.event-band')).toHaveLength(0);
      expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('pending');
      expect(container.querySelector('.forecast-outcome')?.getAttribute('data-onset-seconds')).toBeNull();
      expect(container.querySelector('.forecast-outcome')?.getAttribute('data-resolution-seconds')).toBeNull();
      for (const region of container.querySelectorAll('.forecast-outcome, .event-lane')) { expect(region.textContent).not.toContain('00:25:18'); expect(region.textContent).not.toContain('00:26:17'); }
    }
    const original=container.querySelector('.risk-panel')?.getAttribute('data-value');
    act(()=>store.seek(1577));
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('positive');
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-onset-seconds')).toBe('1518');
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-confirmation-seconds')).toBe('1577');
    expect(container.querySelector('.event-band')?.getAttribute('data-onset-seconds')).toBe('1518');
    expect(screen.getByText('Sustained hypotension occurred within forecast horizon')).toBeTruthy();
    expect(screen.getByText('+00:37')).toBeTruthy();
    expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).toBe(original);
    expect(container.textContent).not.toMatch(/correct prediction|incorrect|true positive|false positive|false negative|true negative|alert fired|model caught|missed event|successful warning/i);
    expect(container.querySelector('.risk-readout')?.className).toBe('risk-readout');
  });
  it('Review immediately permits full historical truth, then Live hides it again',()=>{
    const {container,store}=panels();act(()=>store.seek(1481));expect(screen.getByText('Outcome pending')).toBeTruthy();
    act(()=>store.setMode('review'));
    expect(screen.getByText('Sustained hypotension occurred within forecast horizon')).toBeTruthy();
    expect(screen.getByText('Event details · 1 historical')).toBeTruthy();
    expect(screen.getAllByText('Historical outcome · hindsight, not model input.')).toHaveLength(2);
    act(()=>store.setMode('live'));expect(screen.getByText('Outcome pending')).toBeTruthy();expect(container.querySelectorAll('.event-band')).toHaveLength(0);
  });
  it('shows a frozen negative only at full observation completion, and labels the old horizon without changing its value',()=>{
    const noEvent=outcomeFixture('case-001');const {container,store}=panels(noEvent);
    act(()=>store.seek(15840));expect(screen.getByText('Outcome pending')).toBeTruthy();
    expect(screen.getByText('Forecast horizon ended')).toBeTruthy();expect(screen.getByText(/No newer retained forecast/)).toBeTruthy();
    const original=container.querySelector('.risk-panel')?.getAttribute('data-value');
    act(()=>store.seek(15841));expect(screen.getByText('No sustained hypotension onset occurred within this forecast horizon')).toBeTruthy();
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-resolution-seconds')).toBe('15841');
    expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).toBe(original);
  });
  it('never announces no-event at Live start; Review can state the complete frozen operation result',()=>{
    const {container,store}=panels(outcomeFixture('case-001'));
    expect(container.textContent).not.toContain('No frozen sustained hypotension episodes in this operation.');
    act(()=>store.setMode('review'));expect(screen.getByText('No frozen sustained hypotension episodes in this operation.')).toBeTruthy();
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('not-selected');
  });
  it('supports multiple/recurrent events and labels non-evaluable events without an unsupported forecast outcome',()=>{
    const {container,store}=panels(outcomeFixture('case-007'));act(()=>store.setMode('review'));
    expect(screen.getByText('Event details · 9 historical')).toBeTruthy();
    expect(container.querySelectorAll('.event-details li')).toHaveLength(9);
    expect(screen.getAllByText(/Historical annotation only; not evaluable for forecast outcomes/)).toHaveLength(3);
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('not-selected');
  });
  it('keeps Current MAP score-only beside confirmed truth',()=>{
    const {container,store}=panels(fixture,'current_map');act(()=>store.seek(1577));
    const panel=screen.getByRole('region',{name:'MAP-based ranking score'});
    expect(panel.textContent).not.toMatch(/%|probability|calibrated/i);
    expect(container.querySelector('.risk-panel')?.getAttribute('data-representation')).toBe('raw_score');
    expect(screen.getByText('Sustained hypotension occurred within forecast horizon')).toBeTruthy();
  });
  it('keeps bounded DOM/plot instances and no idle clock loop as events share replay time',()=>{
    const {store,scheduler}=harnessStore();store.seek(1577);store.setSpeed(1);const values=states();let commits=0;
    const {container}=render(<Profiler id="outcomes" onRender={()=>commits++}><RiskHistory modelId="tabpfn_full" store={store} predictions={values.predictions} representation="calibrated"/><EventLane store={store} {...values}/></Profiler>);
    const nodes=container.querySelectorAll('*').length;act(()=>store.play());const before=commits;
    for(let i=0;i<625;i++)act(()=>scheduler.frame(16));
    expect(commits-before).toBeLessThanOrEqual(100);expect(container.querySelectorAll('*')).toHaveLength(nodes);expect(mocks.plots).toHaveLength(1);
    expect(container.querySelector('.event-lane')?.getAttribute('data-permitted-events')).toBe('1');
    act(()=>store.pause());const idle=commits;scheduler.frame(10000);expect(commits).toBe(idle);expect(scheduler.pending.size).toBe(0);
  });
  it('App loads exactly the same-case signals, combined forecasts/truth and events, resetting safely between cases',async()=>{
    const fetchMock=vi.fn(async(path:string)=>{const selected=outcomeFixture(path.includes('case-019')?'case-019':'case-001');return {ok:true,text:async()=>path.endsWith('/events.json')?selected.eventText:path.endsWith('/prediction-windows.json')?selected.text: (await import('./signals.fixture')).signalFixture(selected.selectedCase.ordinal).text};});
    vi.stubGlobal('fetch',fetchMock);render(<App loadData={async()=>metadata}/>);await screen.findByText('Verified monitor traces');
    chooseOption(screen.getByRole('combobox',{name:'Held-out operation'}), 'case-019');
    await screen.findByText('Verified monitor traces');expect((screen.getByRole('slider') as HTMLInputElement).value).toBe('0');
    expect(fetchMock.mock.calls.map(call=>call[0])).toEqual(['case-001','case-019'].flatMap(caseId=>['signals','prediction-windows','events'].map(name=>`/replay-v01/cases/${caseId}/${name}.json`)));
    expect(screen.getByText('No retained forecast at this time')).toBeTruthy();expect(screen.queryByText('Sustained hypotension occurred within forecast horizon')).toBeNull();
  });
});
