import { webcrypto, createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { validateEvents } from '../src/data/eventValidation';
import { loadCaseEvents, eventAssetPath } from '../src/data/caseEvents';
import { loadCaseForecasts, validateForecastOutcomes, forecastState, outcomeState } from '../src/data/forecastOutcomes';
import { validatePredictions } from '../src/data/predictionValidation';
import { historicalContext, visibleEvents, eventBands } from '../src/outcomes/eventVisibility';
import { resolveOutcome } from '../src/outcomes/outcomeResolution';
import { forecastAge } from '../src/outcomes/forecastAge';
import { anchorAt } from '../src/predictions/anchorLookup';
import { forecastValue } from '../src/predictions/representations';
import { historySamples } from '../src/predictions/history';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
import { outcomeFixture } from './outcomes.fixture';
import { metadata } from './metadata.fixture';
const fixture = outcomeFixture();
const anchor = anchorAt(fixture.predictions.windows, 1577)!;
beforeEach(() => vi.stubGlobal('crypto', webcrypto));
afterEach(() => vi.unstubAllGlobals());

describe('frozen events and separately validated historical truth', () => {
  it('validates all 22 immutable event assets and all 3146 outcomes; every positive links to one evaluable frozen event', () => {
    let count = 0; let events = 0;
    for (const selectedCase of metadata.cases) {
      const value = outcomeFixture(selectedCase.ordinal);
      expect(createHash('sha256').update(value.eventText).digest('hex')).toBe(selectedCase.eventAsset.sha256);
      expect(value.events.ordinal).toBe(selectedCase.ordinal); events += value.events.events.length;
      count += value.outcomes.windows.length;
      for (const row of value.outcomes.windows) if (row.label === 1) {
        const matched = value.events.events.filter(event => event.onset_seconds === row.matched_episode_onset_seconds);
        expect(matched).toHaveLength(1); expect(matched[0].evaluable).toBe(true);
      }
      expect(JSON.stringify(value.events)).not.toMatch(/subject_id|case_id|calendar|duration/);
      expect(JSON.stringify(value.predictions)).not.toMatch(/ground_truth|label|onset|episode/);
    }
    expect(count).toBe(3146); expect(events).toBe(43);
  });
  it.each(['subject_id', 'case_id', 'timestamp', 'duration_seconds'])('rejects private, calendar or invented event field %s', field => {
    const raw = structuredClone(fixture.rawEvents); raw[0][field] = 1;
    expect(() => validateEvents(raw, fixture.selectedCase)).toThrow('Historical events unavailable.');
  });
  it('rejects invalid ordinals, times, order, flags, confirmation relationship and invented duration', () => {
    for (const [field, value] of [['presentation_event_ordinal', 0], ['presentation_event_ordinal','event-1'], ['onset_seconds',NaN], ['onset_seconds',1518.5], ['confirmation_seconds',1517], ['confirmation_seconds',1578], ['minimum_confirmed_low_seconds',59], ['sustained_duration_seconds',120], ['evaluable','true'], ['is_recurrent',0]]) {
      const raw = structuredClone(fixture.rawEvents); raw[0][field as string] = value;
      expect(() => validateEvents(raw, fixture.selectedCase)).toThrow();
    }
    const repeated = structuredClone(outcomeFixture('case-003').rawEvents); repeated[1].onset_seconds = repeated[0].onset_seconds;
    expect(() => validateEvents(repeated, fixture.selectedCase)).toThrow();
    expect(() => validateEvents(fixture.rawEvents, {...fixture.selectedCase,durationSeconds:1576})).toThrow();
  });
  it('rejects invalid/ambiguous ground-truth schema and onset outside the frozen horizon', () => {
    for (const truth of [null, {}, {label:2,matched_episode_onset_seconds:null}, {label:0,matched_episode_onset_seconds:1518}, {label:1,matched_episode_onset_seconds:null}, {label:1,matched_episode_onset_seconds:461}, {label:1,matched_episode_onset_seconds:762}, {label:1,matched_episode_onset_seconds:500,subject_id:'forbidden'}]) {
      const raw = structuredClone(fixture.raw); raw[0].ground_truth = truth;
      expect(() => validateForecastOutcomes(raw, fixture.predictions, fixture.selectedCase)).toThrow();
    }
  });
  it('loads only selected-case events with hashes and cancellation; rejects changed bytes/pointers/unsafe ordinals', async () => {
    const fetchMock = vi.fn(async () => ({ok:true,text:async()=>fixture.eventText})); vi.stubGlobal('fetch',fetchMock);
    const controller = new AbortController();
    expect(await loadCaseEvents(fixture.selectedCase,controller.signal)).toEqual(fixture.events);
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith('/replay-v01/cases/case-004/events.json',{signal:controller.signal,cache:'no-cache',redirect:'error'});
    for (const ordinal of ['../case-004','case-023','case-004?x=1']) expect(()=>eventAssetPath(ordinal)).toThrow();
    await expect(loadCaseEvents({...fixture.selectedCase,eventAsset:{...fixture.selectedCase.eventAsset,path:'cases/case-019/events.json'}},controller.signal)).rejects.toThrow();
    fetchMock.mockResolvedValue({ok:true,text:async()=>fixture.eventText+' '});
    await expect(loadCaseEvents(fixture.selectedCase,controller.signal)).rejects.toThrow();
    fetchMock.mockResolvedValue({ok:true,text:async()=>{controller.abort();return fixture.eventText;}});
    await expect(loadCaseEvents(fixture.selectedCase,controller.signal)).rejects.toThrow();
  });
  it('loads forecasts and outcomes once from the same verified bytes, publishing them as separate structures', async () => {
    const fetchMock = vi.fn(async()=>({ok:true,text:async()=>fixture.text}));vi.stubGlobal('fetch',fetchMock);
    const bundle = await loadCaseForecasts(fixture.selectedCase,new AbortController().signal);
    expect(fetchMock).toHaveBeenCalledTimes(1); expect(bundle.forecasts).toEqual(fixture.predictions);
    expect(bundle.outcomes).toEqual(fixture.outcomes);
    expect(forecastState({status:'ready',ordinal:'case-004',data:bundle})).toEqual({status:'ready',ordinal:'case-004',data:fixture.predictions});
    expect(outcomeState({status:'ready',ordinal:'case-004',data:bundle}).status).toBe('ready');
    expect(outcomeState({status:'ready',ordinal:'case-004',data:{...bundle,outcomes:undefined}}).status).toBe('unavailable');
  });
  it('invalid historical truth fails closed independently of unchanged verified forecast values', async () => {
    const raw=structuredClone(fixture.raw);raw[0].ground_truth.label=2;
    const text=JSON.stringify(raw);const selectedCase={...fixture.selectedCase,predictionAsset:{...fixture.selectedCase.predictionAsset,sha256:createHash('sha256').update(text).digest('hex')}};
    vi.stubGlobal('fetch',vi.fn(async()=>({ok:true,text:async()=>text})));
    const bundle=await loadCaseForecasts(selectedCase,new AbortController().signal);
    expect(bundle.forecasts).toEqual(fixture.predictions);expect(bundle.outcomes).toBeUndefined();
    expect(outcomeState({status:'ready',ordinal:'case-004',data:bundle}).status).toBe('unavailable');
  });
  it('retains the poisoned-ground_truth protection in Phase 4 validation', () => {
    const raw = structuredClone(fixture.raw);
    for (const row of raw) Object.defineProperty(row,'ground_truth',{enumerable:true,get:()=>{throw Error('Outcome read');}});
    expect(validatePredictions(raw,fixture.selectedCase)).toEqual(fixture.predictions);
  });
});

describe('causal historical outcomes without detection or decision thresholds', () => {
  it('hides an event before and at onset, before confirmation; reveals it at confirmation with original onset', () => {
    for (const current of [0,1517,1518,1576,1576.99]) expect(visibleEvents(fixture.events,current,'live')).toEqual([]);
    expect(visibleEvents(fixture.events,1577,'live')).toEqual(fixture.events.events);
    expect(visibleEvents(fixture.events,1518,'review')).toEqual(fixture.events.events);
    const band = eventBands(visibleEvents(fixture.events,1577,'live'),[677,1577])[0];
    expect(band.event.onset_seconds).toBe(1518); expect(band.event.confirmation_seconds).toBe(1577);
    expect(band.width).toBeCloseTo(59/900*100);
  });
  it('exposes no positive onset, label or resolution time before confirmation; resolves at exact confirmation', () => {
    for (const current of [1481,1518,1576.99]) expect(resolveOutcome(anchor,fixture.outcomes,fixture.events,current,'live')).toEqual({status:'pending'});
    expect(resolveOutcome(anchor,fixture.outcomes,fixture.events,1577,'live')).toEqual({status:'positive',event:fixture.events.events[0],timeAfterForecast:37,resolutionSeconds:1577});
    expect(resolveOutcome(anchor,fixture.outcomes,fixture.events,1481,'review').status).toBe('positive');
  });
  it('negative remains pending through the full observation boundary; Review can show the frozen label immediately', () => {
    const negative = fixture.predictions.windows[0];
    expect(resolveOutcome(negative,fixture.outcomes,fixture.events,760,'live')).toEqual({status:'pending'});
    expect(resolveOutcome(negative,fixture.outcomes,fixture.events,820.99,'live')).toEqual({status:'pending'});
    expect(resolveOutcome(negative,fixture.outcomes,fixture.events,821,'live')).toEqual({status:'negative',resolutionSeconds:821});
    expect(resolveOutcome(negative,fixture.outcomes,fixture.events,461,'review')).toEqual({status:'negative',resolutionSeconds:821});
  });
  it('fails closed on missing/duplicate/non-evaluable event links or mismatched outcome identities', () => {
    for (const events of [{...fixture.events,events:[]},{...fixture.events,events:[...fixture.events.events,...fixture.events.events]},{...fixture.events,events:[{...fixture.events.events[0],evaluable:false}]}]) {
      expect(resolveOutcome(anchor,fixture.outcomes,events,2000,'review')).toEqual({status:'unavailable'});
    }
    expect(resolveOutcome(anchor,{...fixture.outcomes,ordinal:'case-019'},fixture.events,2000,'live')).toEqual({status:'unavailable'});
    expect(resolveOutcome(anchor,{...fixture.outcomes,windows:[]},fixture.events,2000,'live')).toEqual({status:'unavailable'});
  });
  it('supports recurrent and non-evaluable annotations without inventing forecast linkage', () => {
    const multiple=outcomeFixture('case-007');
    expect(visibleEvents(multiple.events,0,'review')).toHaveLength(9);
    expect(multiple.events.events.filter(event=>event.is_recurrent)).toHaveLength(8);
    const notEvaluable=multiple.events.events[3];expect(notEvaluable.evaluable).toBe(false);
    expect(visibleEvents(multiple.events,notEvaluable.confirmation_seconds-1,'live')).not.toContain(notEvaluable);
    expect(visibleEvents(multiple.events,notEvaluable.confirmation_seconds,'live')).toContain(notEvaluable);
    expect(outcomeFixture('case-001').events.events).toEqual([]);
  });
  it('requires matching verified signal, forecast, truth and event sessions', () => {
    const store=new ReplayStore('case-004',7500,new ManualScheduler());
    const predictions={status:'ready' as const,ordinal:'case-004' as const,data:fixture.predictions};
    const outcomes={status:'ready' as const,ordinal:'case-004' as const,data:fixture.outcomes};
    const events={status:'ready' as const,ordinal:'case-004' as const,data:fixture.events};
    expect(historicalContext(events,outcomes,predictions,store.getSnapshot())).toBeUndefined();store.activate();
    expect(historicalContext(events,outcomes,predictions,store.getSnapshot())).toBeDefined();
    expect(historicalContext({...events,ordinal:'case-019'},outcomes,predictions,store.getSnapshot())).toBeUndefined();
    expect(historicalContext(events,{...outcomes,data:{...fixture.outcomes,ordinal:'case-019' as const}},predictions,store.getSnapshot())).toBeUndefined();
  });
  it('labels original horizon expiry without changing anchor, value, calibration or point colors/data', () => {
    const row=anchorAt(fixture.predictions.windows,1600)!;const before=JSON.stringify(fixture.predictions);
    expect(forecastAge(row,1781)).toEqual({expired:false,title:'Forecast age',value:'300 s'});
    expect(forecastAge(row,1907)).toEqual({expired:true,title:'Forecast horizon ended',value:'2m 06s ago'});
    expect(row.anchor_seconds).toBe(1481);expect(row.horizon_end_inclusive_seconds).toBe(1781);
    const samples=historySamples(fixture.predictions.windows,0,20,'tabpfn_full','calibrated');
    const value=forecastValue(row,'tabpfn_full');resolveOutcome(row,fixture.outcomes,fixture.events,1907,'review');
    expect(historySamples(fixture.predictions.windows,0,20,'tabpfn_full','calibrated')).toEqual(samples);
    expect(forecastValue(row,'tabpfn_full')).toEqual(value);expect(JSON.stringify(fixture.predictions)).toBe(before);
    expect(readFileSync('src/charts/riskOptions.ts','utf8')).not.toMatch(/outcome|event|crimson|#bc858e/);
    for(const file of ['src/outcomes/outcomeResolution.ts','src/outcomes/eventVisibility.ts','src/data/eventValidation.ts','src/data/forecastOutcomes.ts']) {
      expect(readFileSync(file,'utf8')).not.toMatch(/channels\.map|map\.values|sklearn|predict_proba|LogisticRegression|Platt/);
    }
  });
});
