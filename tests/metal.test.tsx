import { chooseOption } from './select.fixture';
import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { argentConfig, fillConfig, paperUniforms, pixelBudget } from '../src/metal/argentConfig';
import { metalEnvironment, permitsPaper } from '../src/metal/capability';
import { SurfaceBudget, surfaceBudget } from '../src/metal/lifecycle';
import { InteractiveMetalTab } from '../src/metal/InteractiveMetalTab';
import { PaperSurface } from '../src/metal/PaperSurface';
import { MercuryReplaySwitch } from '../src/metal/MercuryReplaySwitch';
import { MoltenProbabilityProgress } from '../src/metal/MoltenProbabilityProgress';
import { RiskPanel } from '../src/components/RiskPanel';
import { ModelSelector } from '../src/components/ModelSelector';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
import { outcomeFixture } from './outcomes.fixture';
import { metadata } from './metadata.fixture';

const mock = vi.hoisted(() => ({ fail: false, mounts: [] as { dispose: ReturnType<typeof vi.fn>; canvasElement: HTMLCanvasElement; setMaxPixelCount: ReturnType<typeof vi.fn> }[] }));
vi.mock('@paper-design/shaders', async importOriginal => ({ ...(await importOriginal<Record<string,unknown>>()),
  ShaderMount: class {
    canvasElement=document.createElement('canvas');setMaxPixelCount=vi.fn();dispose=vi.fn(()=>this.canvasElement.remove());
    constructor(parent:HTMLElement) { parent.append(this.canvasElement);if(mock.fail) throw new Error('Synthetic WebGL initialization failure');mock.mounts.push(this); }
  }
}));
class IntersectionMock {
  static entries:IntersectionMock[]=[];
  constructor(public callback:IntersectionObserverCallback) { IntersectionMock.entries.push(this); }
  observe(target:Element) { this.callback([{target,isIntersecting:true} as IntersectionObserverEntry],this as unknown as IntersectionObserver); }
  disconnect=vi.fn();
}
class ResizeMock {
  observe=vi.fn();disconnect=vi.fn();
}
let hidden=false;let reduced=false;let mediaChange:()=>void;
const loseContext=vi.fn();
beforeEach(()=>{
  mock.fail=false;mock.mounts=[];IntersectionMock.entries=[];hidden=false;reduced=false;
  vi.stubGlobal('WebGL2RenderingContext',class {});vi.stubGlobal('ResizeObserver',ResizeMock);vi.stubGlobal('IntersectionObserver',IntersectionMock);
  vi.spyOn(document,'hidden','get').mockImplementation(()=>hidden);
  vi.spyOn(window,'matchMedia').mockImplementation(query=>({matches:reduced,media:query,addEventListener:(_name:string,handler:()=>void)=>{mediaChange=handler;},removeEventListener:vi.fn()} as unknown as MediaQueryList));
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockImplementation(()=>({isContextLost:()=>false,getExtension:()=>({loseContext})}) as unknown as WebGLRenderingContext);
});
afterEach(()=>{cleanup();vi.unstubAllGlobals();window.history.replaceState(null,'','/');expect(surfaceBudget.counts().total).toBe(0);});
const fixture=outcomeFixture();
function panel(modelId='tabpfn_full',value=fixture) {
  const scheduler=new ManualScheduler();const store=new ReplayStore(value.selectedCase.ordinal,value.selectedCase.durationSeconds,scheduler);store.activate();
  const props={model:metadata.models.find(m=>m.id===modelId)!,store,
    predictions:{status:'ready' as const,ordinal:value.selectedCase.ordinal,data:value.predictions},
    outcomes:{status:'ready' as const,ordinal:value.selectedCase.ordinal,data:value.outcomes},
    events:{status:'ready' as const,ordinal:value.selectedCase.ordinal,data:value.events},onRepresentationChange:vi.fn()};
  return {store,scheduler,props};
}
describe('audited Argent/Paper contract',()=>{
  it('maps only gunmetal, border/single/surface, Paper and exact settings',()=>{
    expect(argentConfig).toEqual({tone:'gunmetal',variant:'border',frame:'single',finish:'surface',engine:'paper',radius:20,speed:1.5,metalScale:1,angle:30,sheen:true,revealOnHover:true});
    expect(fillConfig).toEqual({tone:'gunmetal',finish:'surface',engine:'paper',speed:1.5,scale:1,angle:30});
    expect(paperUniforms()).toMatchObject({u_angle:30,u_scale:1,u_isImage:false,u_rotation:0});
    expect(Object.values(paperUniforms()).some(v=>typeof v==='string')).toBe(false);
  });
  it('pins new versions and preserves all existing dependency versions',()=>{
    const pkg=JSON.parse(readFileSync('package.json','utf8'));expect(pkg.dependencies).toEqual({'@paper-design/shaders':'0.0.81','@paper-design/shaders-react':'0.0.81',argentui:'0.4.1',react:'19.3.0','react-dom':'19.3.0',uplot:'1.6.32'});
    expect(readFileSync('THIRD_PARTY_NOTICES.md','utf8')).toContain('Apache-2.0');expect(readFileSync('THIRD_PARTY_NOTICES.md','utf8')).toContain('Copyright 2026 Paper');
  });
  it('caps physical pixel area to 1.5 DPR without an unrestricted Retina default',()=>{
    expect(pixelBudget(200,10)).toBe(4500);expect(pixelBudget(0,0)).toBe(1);
  });
  it('has no native, remote, scientific inference or outcome-dependent metal path',()=>{
    for(const name of readdirSync('src/metal').filter(name=>/tsx?$/.test(name))) {
      const source=readFileSync(`src/metal/${name}`,'utf8');expect(source).not.toMatch(/engine:\s*'native'|NATIVE_TONES|fetch\(|https?:|\.fit\(|predict_proba|ground_truth|matched_episode|historical-positive|outcomeResolution/);
    }
  });
  it('keeps scientific selectors, loaders, plot options, clock, outcome and event style hashes unchanged; interaction adapters are explicitly authorized',()=>{
    const checks=JSON.parse(readFileSync('tests/fixtures/phase5-source-hashes.json','utf8')).files.sha256;
    for(const [name,hash] of Object.entries(checks)) {
      if(name.startsWith('src/data/')||name.startsWith('src/outcomes/')||['src/components/EventLane.tsx','src/components/ForecastOutcome.tsx','src/styles/shell.css'].includes(name)) expect(createHash('sha256').update(readFileSync(name)).digest('hex')).toBe(hash);
    }
    const older=JSON.parse(readFileSync('tests/fixtures/phase4-source-hashes.json','utf8')).files.sha256;
    for(const name of ['src/predictions/anchorLookup.ts','src/predictions/representations.ts','src/predictions/history.ts','src/charts/RiskHistoryPlot.tsx','src/charts/riskOptions.ts','src/components/RiskHistory.tsx']) expect(createHash('sha256').update(readFileSync(name)).digest('hex')).toBe(older[name]);
    const phase3=JSON.parse(readFileSync('tests/fixtures/phase3-source-hashes.json','utf8')).files.sha256;
    for(const name of ['src/replay/clock.ts','src/replay/viewport.ts','src/replay/visibility.ts']) expect(createHash('sha256').update(readFileSync(name)).digest('hex')).toBe(phase3[name]);
  });
});
describe('interactive gunmetal controls',()=>{
  it('idle is border-dominant with no transient canvas; hover and leave mount/dispose exactly one',()=>{
    const {container}=render(<InteractiveMetalTab><button>Random</button></InteractiveMetalTab>);const shell=container.querySelector('.metal-tab')!;
    expect(shell.getAttribute('data-variant')).toBe('border');expect(container.querySelector('canvas')).toBeNull();
    fireEvent.pointerEnter(shell);expect(container.querySelectorAll('[data-metal-canvas="transient"]')).toHaveLength(1);
    expect(shell.getAttribute('data-engaged')).toBe('true');fireEvent.pointerLeave(shell);expect(container.querySelector('canvas')).toBeNull();expect(shell.getAttribute('data-engaged')).toBeNull();
  });
  it('keyboard focus enables fill and blur releases it without changing native behavior',()=>{
    const click=vi.fn();const {container}=render(<InteractiveMetalTab><button onClick={click}>Play</button></InteractiveMetalTab>);
    fireEvent.focus(screen.getByRole('button'));expect(container.querySelector('canvas')).not.toBeNull();fireEvent.click(screen.getByRole('button'));expect(click).toHaveBeenCalledOnce();
    fireEvent.blur(screen.getByRole('button'));expect(container.querySelector('canvas')).toBeNull();
  });
  it('disabled controls never activate a shader or fill',()=>{
    const {container,rerender}=render(<InteractiveMetalTab><button disabled>Previous</button></InteractiveMetalTab>);const shell=container.querySelector('.metal-tab')!;
    fireEvent.pointerEnter(shell);fireEvent.focus(screen.getByRole('button'));expect(container.querySelector('canvas')).toBeNull();expect(shell.getAttribute('data-engaged')).toBeNull();
    rerender(<InteractiveMetalTab><button>Previous</button></InteractiveMetalTab>);fireEvent.pointerEnter(shell);expect(container.querySelector('canvas')).not.toBeNull();
    rerender(<InteractiveMetalTab><button disabled>Previous</button></InteractiveMetalTab>);expect(container.querySelector('canvas')).toBeNull();
  });
  it('selected controls keep a static indicator without permanent fill',()=>{
    const {container}=render(<InteractiveMetalTab active><button>Pause</button></InteractiveMetalTab>);
    expect(container.querySelector('.metal-tab')?.getAttribute('data-active')).toBe('true');expect(container.querySelector('.metal-tab')?.getAttribute('data-engaged')).toBeNull();expect(container.querySelector('canvas')).toBeNull();
  });
  it('arbitrates keyboard focus plus another hovered pill to at most one transient',()=>{
    const {container}=render(<><InteractiveMetalTab><button>One</button></InteractiveMetalTab><InteractiveMetalTab><button>Two</button></InteractiveMetalTab></>);
    fireEvent.focus(screen.getByText('One'));fireEvent.pointerEnter(screen.getByText('Two').parentElement!);
    expect(container.querySelectorAll('canvas')).toHaveLength(1);expect(surfaceBudget.counts().transient).toBe(1);
  });
  it('retains select-only combobox semantics and uses an 18px right-inset chevron',()=>{
    const change=vi.fn();const {container}=render(<ModelSelector models={metadata.models} value="logistic_map" onChange={change}/>);
    chooseOption(screen.getByRole('combobox'), 'tabpfn_full');expect(change).toHaveBeenCalledWith('tabpfn_full');expect(container.querySelector('.metal-select-chevron')).not.toBeNull();
    const css=readFileSync('src/metal/metal.css','utf8');expect(css).toMatch(/\.metal-select-chevron[^}]*right: 18px/);expect(css).toContain('padding-right: 44px');
  });
  it('Mercury changes only the controlled two-state mode, preserving radio labels',()=>{
    const change=vi.fn();const {container,rerender}=render(<MercuryReplaySwitch value="live" onChange={change}/>);
    expect((screen.getByRole('radio',{name:'Live Replay'}) as HTMLInputElement).checked).toBe(true);fireEvent.click(screen.getByRole('radio',{name:'Review'}));expect(change).toHaveBeenCalledWith('review');
    rerender(<MercuryReplaySwitch value="review" onChange={change}/>);expect((screen.getByRole('radio',{name:'Review'}) as HTMLInputElement).checked).toBe(true);expect(container.querySelector('.mercury-switch')?.getAttribute('data-mode')).toBe('review');expect(surfaceBudget.counts().persistent).toBe(1);
  });
});
describe('molten geometry preserves retained scientific values',()=>{
  it.each([.0022565323804585326,.009513229723575736,.3038048376358691,.374864114478624])('retains exact probability %s with direct geometry and accessible percent',probability=>{
    const {container}=render(<MoltenProbabilityProgress probability={probability} representation="calibrated_probability"/>);
    expect(container.querySelector('.molten-progress')?.getAttribute('data-probability')).toBe(String(probability));expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe(String(probability*100));
    expect(container.querySelector<HTMLElement>('.molten-fill')?.style.clipPath).toBe(`inset(0 ${100-probability*100}% 0 0)`);
  });
  it('unavailable is an inactive shell, without role, percent or fake zero',()=>{
    const {container}=render(<MoltenProbabilityProgress representation="raw_probability"/>);
    expect(screen.queryByRole('progressbar')).toBeNull();expect(container.querySelector('.molten-progress')?.getAttribute('data-probability')).toBeNull();expect(container.querySelector('canvas')).toBeNull();
  });
  it('Current MAP and prevalence never receive Molten Progress',()=>{
    const a=panel('current_map');a.store.seek(1577);const {container,rerender}=render(<RiskPanel {...a.props} representation="calibrated"/>);expect(container.querySelector('.molten-progress')).toBeNull();expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).toBe('-80');
    const b=panel('prevalence');b.store.seek(1577);rerender(<RiskPanel {...b.props} representation="calibrated"/>);expect(container.querySelector('.molten-progress')).toBeNull();
  });
  it('selects raw/calibrated without recomputation and preserves positive resolution/horizon styling',()=>{
    const {store,props}=panel();store.seek(1518);const {container,rerender}=render(<RiskPanel {...props} representation="calibrated"/>);
    const before=container.querySelector('.molten-progress')?.getAttribute('data-probability');expect(before).toBe('0.009513229723575736');expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('pending');
    act(()=>store.seek(1577));expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('positive');expect(container.querySelector('.molten-progress')?.getAttribute('data-probability')).toBe(before);
    expect(container.querySelector('.molten-progress')?.getAttribute('data-tone')).toBe('gunmetal');expect(container.querySelector('.risk-panel')?.getAttribute('data-horizon-end')).toBe('1781');
    rerender(<RiskPanel {...props} representation="raw"/>);expect(container.querySelector('.molten-progress')?.getAttribute('data-probability')).toBe('0.0126526253297925');
  });
  it('expiry retains exact geometry and does not change a negative outcome early',()=>{
    const {store,props}=panel('tabpfn_full',outcomeFixture('case-001'));store.seek(15840);const {container}=render(<RiskPanel {...props} representation="calibrated"/>);
    expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('pending');expect(container.textContent).toContain('Forecast horizon ended');const before=container.querySelector('.molten-fill')?.getAttribute('style');
    act(()=>store.seek(15841));expect(container.querySelector('.forecast-outcome')?.getAttribute('data-outcome-state')).toBe('negative');expect(container.querySelector('.molten-fill')?.getAttribute('style')).toBe(before);
  });
});
describe('bounded Paper lifecycle and failure fallback',()=>{
  it('enforces two persistent and one transient mount; replacement is disposed synchronously',()=>{
    const budget=new SurfaceBudget();const stop=vi.fn();budget.acquire('mercury','m',stop);budget.acquire('probability','p',stop);budget.acquire('transient','t',stop);expect(budget.counts()).toEqual({persistent:2,transient:1,total:3});
    budget.acquire('transient','u',stop);expect(stop).toHaveBeenCalledOnce();expect(budget.counts().total).toBe(3);
  });
  it('three persistent surfaces plus one transient maximum, idle pills use no context',()=>{
    const {container}=render(<><MercuryReplaySwitch value="live" onChange={()=>{}}/><MoltenProbabilityProgress probability={.3} representation="raw_probability"/><PaperSurface kind="timeline"/><InteractiveMetalTab><button>Hover</button></InteractiveMetalTab></>);
    expect(container.querySelectorAll('canvas')).toHaveLength(3);fireEvent.pointerEnter(screen.getByText('Hover').parentElement!);expect(container.querySelectorAll('canvas')).toHaveLength(4);expect(surfaceBudget.counts()).toEqual({persistent:3,transient:1,total:4});fireEvent.pointerLeave(screen.getByText('Hover').parentElement!);expect(container.querySelectorAll('canvas')).toHaveLength(3);
  });
  it('initialization failure catches locally, releases the lease and preserves static geometry',()=>{
    mock.fail=true;const {container}=render(<MoltenProbabilityProgress probability={.304} representation="raw_probability"/>);expect(container.querySelector('canvas')).toBeNull();expect(container.querySelector('.paper-surface')?.getAttribute('data-context-failed')).toBe('true');expect(container.querySelector('.molten-progress')?.getAttribute('data-probability')).toBe('0.304');expect(surfaceBudget.counts().total).toBe(0);
  });
  it('context loss latches static fallback without resetting replay or retrying on restoration',()=>{
    const {store,props}=panel();store.seek(1577);const snapshot=store.getSnapshot();const {container}=render(<RiskPanel {...props} representation="calibrated"/>);const canvas=container.querySelector('canvas')!;
    fireEvent(canvas,new Event('webglcontextlost',{bubbles:true,cancelable:true}));expect(container.querySelector('canvas')).toBeNull();expect(store.getSnapshot()).toBe(snapshot);
    fireEvent(canvas,new Event('webglcontextrestored'));expect(mock.mounts).toHaveLength(1);expect(container.querySelector('.molten-progress')?.getAttribute('data-probability')).toBe('0.009513229723575736');
  });
  it('reduced motion uses static gunmetal with no recurring shader work',()=>{
    reduced=true;const {container}=render(<PaperSurface kind="mercury"/>);expect(container.querySelector('canvas')).toBeNull();expect(container.querySelector('.paper-surface')?.getAttribute('data-metal-motion')).toBe('static');
  });
  it('media preference changes suspend an existing shader',()=>{
    const {container}=render(<PaperSurface kind="mercury"/>);expect(container.querySelector('canvas')).not.toBeNull();reduced=true;act(()=>mediaChange());expect(container.querySelector('canvas')).toBeNull();
  });
  it('document hidden releases the shader and resuming does not change replay state',()=>{
    const {container}=render(<PaperSurface kind="mercury"/>);expect(container.querySelector('canvas')).not.toBeNull();hidden=true;fireEvent(document,new Event('visibilitychange'));expect(container.querySelector('canvas')).toBeNull();
    hidden=false;fireEvent(document,new Event('visibilitychange'));expect(container.querySelector('canvas')).not.toBeNull();
  });
  it('offscreen surfaces unmount and release context, then reenter safely',()=>{
    const {container}=render(<PaperSurface kind="mercury"/>);const observer=IntersectionMock.entries[0];
    act(()=>observer.callback([{isIntersecting:false} as IntersectionObserverEntry],observer as unknown as IntersectionObserver));expect(container.querySelector('canvas')).toBeNull();
    act(()=>observer.callback([{isIntersecting:true} as IntersectionObserverEntry],observer as unknown as IntersectionObserver));expect(container.querySelector('canvas')).not.toBeNull();
  });
  it('forced static capability leaves controls working without WebGL',()=>{
    window.history.replaceState(null,'','/?metal=static');expect(permitsPaper(metalEnvironment(),true)).toBe(false);
    const change=vi.fn();const {container}=render(<MercuryReplaySwitch value="live" onChange={change}/>);fireEvent.click(screen.getByRole('radio',{name:'Review'}));expect(change).toHaveBeenCalledWith('review');expect(container.querySelector('canvas')).toBeNull();
  });
});
