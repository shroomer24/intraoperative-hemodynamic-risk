import { readFileSync } from 'node:fs';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { GunmetalSelect } from '../src/metal/GunmetalSelect';
import { ModelSelector } from '../src/components/ModelSelector';
import { CaseSelector } from '../src/components/CaseSelector';
import { playbackSpeeds } from '../src/replay/clock';
import { metadata } from './metadata.fixture';
const options=[{value:'a',label:'Alpha'},{value:'b',label:'Beta',disabled:true},{value:'c',label:'Charlie'},{value:'d',label:'Delta',group:'Technical'}];
function setup(){const change=vi.fn();const result=render(<GunmetalSelect id="demo" label="Demo" value="c" options={options} onChange={change}/>);return {...result,change,control:screen.getByRole('combobox')};}
describe('dark select-only listbox',()=>{
  it('has one native button focus target, with no native select/popup route',()=>{
    const {container,control}=setup();expect(control.tagName).toBe('BUTTON');expect(control.getAttribute('aria-expanded')).toBe('false');expect(container.querySelector('select')).toBeNull();
    for(const file of ['ModelSelector','CaseSelector','RiskPanel','PlaybackControls'])expect(readFileSync(`src/components/${file}.tsx`,'utf8')).not.toMatch(/<select|<option|<optgroup/);
    expect(container.querySelector('.metal-select-chevron')).not.toBeNull();
  });
  it('opens with Enter, preserves selected option and exposes active descendant',()=>{
    const {control}=setup();fireEvent.keyDown(control,{key:'Enter'});expect(control.getAttribute('aria-expanded')).toBe('true');
    const selected=screen.getByRole('option',{name:'Charlie'});expect(selected.getAttribute('aria-selected')).toBe('true');expect(control.getAttribute('aria-activedescendant')).toBe(selected.id);
    expect(screen.getByRole('listbox').id).toBe(control.getAttribute('aria-controls'));expect(screen.getByRole('group',{name:'Technical'})).toBeTruthy();
    expect(screen.queryByRole('option',{name:'Technical'})).toBeNull();expect(screen.getByRole('option',{name:'Beta'}).getAttribute('aria-disabled')).toBe('true');
  });
  it('Space opens; arrows skip disabled options; Home/End navigate; Enter commits and restores focus',()=>{
    const {control,change}=setup();fireEvent.keyDown(control,{key:' '});fireEvent.keyDown(control,{key:'Home'});expect(control.getAttribute('aria-activedescendant')).toBe(screen.getByRole('option',{name:'Alpha'}).id);
    fireEvent.keyDown(control,{key:'ArrowDown'});expect(control.getAttribute('aria-activedescendant')).toBe(screen.getByRole('option',{name:'Charlie'}).id);
    fireEvent.keyDown(control,{key:'End'});expect(control.getAttribute('aria-activedescendant')).toBe(screen.getByRole('option',{name:'Delta'}).id);
    fireEvent.keyDown(control,{key:'ArrowUp'});fireEvent.keyDown(control,{key:'Enter'});expect(change).toHaveBeenCalledWith('c');expect(screen.queryByRole('listbox')).toBeNull();expect(document.activeElement).toBe(control);
  });
  it('Escape closes without committing the exploratory active option',()=>{
    const {control,change}=setup();fireEvent.click(control);fireEvent.keyDown(control,{key:'Home'});fireEvent.keyDown(control,{key:'Escape'});expect(change).not.toHaveBeenCalled();expect(control.getAttribute('data-value')).toBe('c');expect(screen.queryByRole('listbox')).toBeNull();
  });
  it('Tab closes without preventing normal focus exit',()=>{
    const {control}=setup();fireEvent.click(control);expect(fireEvent.keyDown(control,{key:'Tab'})).toBe(true);expect(screen.queryByRole('listbox')).toBeNull();
  });
  it('outside pointer closes; option pointer selects and restores trigger focus',()=>{
    const {control,change}=setup();fireEvent.click(control);fireEvent.pointerDown(document.body);expect(screen.queryByRole('listbox')).toBeNull();fireEvent.click(control);fireEvent.click(screen.getByRole('option',{name:'Alpha'}));expect(change).toHaveBeenCalledWith('a');expect(document.activeElement).toBe(control);
  });
  it('typeahead activates a matching label without changing the committed value',()=>{
    const {control,change}=setup();fireEvent.keyDown(control,{key:'d'});expect(control.getAttribute('aria-activedescendant')).toBe(screen.getByRole('option',{name:'Delta'}).id);expect(change).not.toHaveBeenCalled();
  });
  it('disabled trigger/options and group headings cannot select',()=>{
    const {control,change,rerender}=setup();fireEvent.click(control);fireEvent.click(screen.getByRole('option',{name:'Beta'}));expect(change).not.toHaveBeenCalled();
    rerender(<GunmetalSelect label="Demo" value="c" options={options} onChange={change} disabled/>);expect((screen.getByRole('combobox') as HTMLButtonElement).disabled).toBe(true);expect(screen.queryByRole('listbox')).toBeNull();
  });
  it('retains exact model order/labels and baseline semantic grouping',()=>{
    render(<ModelSelector models={metadata.models} value="logistic_map" onChange={()=>{}}/>);fireEvent.click(screen.getByRole('combobox'));expect(screen.getAllByRole('option').map(o=>o.getAttribute('data-value'))).toEqual(['current_map','logistic_map','logistic_full','xgboost','tabpfn_map','tabpfn_full','prevalence']);expect(screen.getByRole('group',{name:'Baseline / technical'})).toBeTruthy();
  });
  it('retains exactly 22 cases and scrollable selected-option handling',()=>{
    render(<CaseSelector cases={metadata.cases} value="case-022" onChange={()=>{}} onRandom={()=>{}}/>);fireEvent.click(screen.getByRole('combobox'));const list=screen.getByRole('listbox');expect(within(list).getAllByRole('option')).toHaveLength(22);expect(screen.getByRole('option',{name:'Case 022'}).getAttribute('aria-selected')).toBe('true');
    expect(readFileSync('src/metal/GunmetalSelect.tsx','utf8')).toContain('menu.scrollTop');
  });
  it('uses dark CSS menus/rows, 18px chevrons, and no menu-row Paper surfaces',()=>{
    const css=readFileSync('src/styles/interaction.css','utf8');expect(css).toContain('background: #12151a');expect(css).toContain('overflow-y: auto');expect(css).toContain('border-radius: 16px');expect(readFileSync('src/metal/metal.css','utf8')).toMatch(/\.metal-select-chevron[^}]*right: 18px/);
    expect(readFileSync('src/metal/GunmetalSelect.tsx','utf8')).not.toMatch(/<PaperSurface|<select|<option/);expect(playbackSpeeds).toEqual([1,10,30,60]);
    expect(readFileSync('src/components/RiskPanel.tsx','utf8')).toContain("{value:'calibrated',label:'Calibrated'},{value:'raw',label:'Retained raw'}");
  });
});
