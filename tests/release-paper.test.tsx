import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { PaperSurface } from '../src/metal/PaperSurface';
import { ReplayMetalMotion, METAL_SETTLE_MS } from '../src/metal/ReplayMetalMotion';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
const record=vi.hoisted(()=>({mounts:[] as {setSpeed:ReturnType<typeof vi.fn>;dispose:ReturnType<typeof vi.fn>}[]}));
vi.mock('@paper-design/shaders',async original=>({...await original<Record<string,unknown>>(),ShaderMount:class{
  canvasElement=document.createElement('canvas');setSpeed=vi.fn();setMaxPixelCount=vi.fn();dispose=vi.fn(()=>this.canvasElement.remove());
  constructor(parent:HTMLElement){parent.append(this.canvasElement);record.mounts.push(this);}
}}));
beforeEach(()=>{
  vi.useFakeTimers();record.mounts=[];vi.stubGlobal('WebGL2RenderingContext',class{});
  vi.stubGlobal('IntersectionObserver',class{constructor(private callback:IntersectionObserverCallback){} observe(target:Element){this.callback([{target,isIntersecting:true} as IntersectionObserverEntry],this as unknown as IntersectionObserver);}disconnect(){}});
  vi.stubGlobal('ResizeObserver',class{observe(){}disconnect(){}});
  vi.spyOn(window,'matchMedia').mockImplementation(query=>({matches:false,media:query,addEventListener(){},removeEventListener(){}} as unknown as MediaQueryList));
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockImplementation(()=>({isContextLost:()=>false,getExtension:()=>null}) as unknown as WebGLRenderingContext);
});
afterEach(()=>{cleanup();vi.unstubAllGlobals();vi.useRealTimers();});
it('stops the actual shader adapter while idle, resumes for playback and settles input without constructing another context',()=>{
 const scheduler=new ManualScheduler();const store=new ReplayStore('case-019',2100,scheduler);store.activate();
 render(<ReplayMetalMotion store={store}><PaperSurface kind="mercury"/></ReplayMetalMotion>);
 expect(record.mounts).toHaveLength(1);const mount=record.mounts[0];expect(mount.setSpeed).toHaveBeenLastCalledWith(0);
 act(()=>store.play());expect(mount.setSpeed).toHaveBeenLastCalledWith(1.5);
 act(()=>store.pause());expect(mount.setSpeed).toHaveBeenLastCalledWith(0);
 act(()=>window.dispatchEvent(new Event('pointerdown')));expect(mount.setSpeed).toHaveBeenLastCalledWith(1.5);
 act(()=>vi.advanceTimersByTime(METAL_SETTLE_MS));expect(mount.setSpeed).toHaveBeenLastCalledWith(0);
 expect(record.mounts).toHaveLength(1);expect(mount.dispose).not.toHaveBeenCalled();store.dispose();
});

it('keeps an engaged transient shader alive beyond five seconds, releases it on leave and restarts without leaks',()=>{
 const store=new ReplayStore('case-019',2100,new ManualScheduler());store.activate();
 const view=render(<ReplayMetalMotion store={store}><PaperSurface kind="transient"/></ReplayMetalMotion>);
 for(let cycle=0;cycle<3;cycle++){
   if(cycle) view.rerender(<ReplayMetalMotion store={store}><PaperSurface kind="transient"/></ReplayMetalMotion>);
   const mount=record.mounts[cycle];expect(mount.setSpeed).toHaveBeenLastCalledWith(1.5);
   act(()=>vi.advanceTimersByTime(6000));expect(mount.setSpeed).toHaveBeenLastCalledWith(1.5);expect(mount.dispose).not.toHaveBeenCalled();
   view.rerender(<ReplayMetalMotion store={store}><span/></ReplayMetalMotion>);expect(mount.dispose).toHaveBeenCalledTimes(1);
   expect(document.querySelectorAll('canvas')).toHaveLength(0);
 }
 store.dispose();
});
