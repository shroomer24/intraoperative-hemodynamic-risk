import { GunmetalSelect } from '../metal/GunmetalSelect';
import { modelLabels } from '../data/types';
import type { ModelDefinition, ModelId } from '../data/types';

const mainModels: ModelId[] = ['current_map', 'logistic_map', 'logistic_full', 'xgboost', 'tabpfn_map', 'tabpfn_full'];
export function ModelSelector({ models, value, onChange }: {
  models: ModelDefinition[]; value: ModelId; onChange: (id: ModelId) => void;
}) {
  const selected = models.find(model => model.id === value)!;
  return <div className="selector-field model-field">
    <label htmlFor="model-select" className="field-label">Model</label>
    <GunmetalSelect id="model-select" label="Model" value={value} onChange={next => onChange(next as ModelId)}
      options={[...mainModels.filter(id => models.some(model => model.id === id)).map(id => ({value:id,label:modelLabels[id]})),
        {value:'prevalence',label:modelLabels.prevalence,group:'Baseline / technical'}]} />
    <span className="field-note">{selected.featureCount === 0 ? 'Fixed TRAINING baseline' : `${selected.featureCount} ${selected.featureCount === 1 ? 'predictor' : 'predictors'}`} · frozen specification</span>
  </div>;
}
