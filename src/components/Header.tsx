import type { CaseOrdinal, ModelId, PresentationCase, ReplayMode, ShellMetadata } from '../data/types';
import { FrozenStudyLabel } from './FrozenStudyLabel';
import { ModelSelector } from './ModelSelector';
import { CaseSelector } from './CaseSelector';
import { ReplayModeSelector } from './ReplayModeSelector';
import { ProjectTitle } from './ProjectTitle';

interface Props {
  metadata: ShellMetadata;
  modelId: ModelId;
  onModelChange: (id: ModelId) => void;
  selectedCase: PresentationCase;
  onCaseChange: (ordinal: CaseOrdinal) => void;
  onRandomCase: () => void;
  mode: ReplayMode;
  onModeChange: (mode: ReplayMode) => void;
}
export function Header(props: Props) {
  return <header className="app-header">
    <div className="title-row">
      <ProjectTitle />
      <div className="study-status"><FrozenStudyLabel /><span className="shell-label">Historical replay</span></div>
    </div>
    <div className="selection-bar">
      <ModelSelector models={props.metadata.models} value={props.modelId} onChange={props.onModelChange} />
      <CaseSelector cases={props.metadata.cases} value={props.selectedCase.ordinal}
        onChange={props.onCaseChange} onRandom={props.onRandomCase} />
      <ReplayModeSelector value={props.mode} onChange={props.onModeChange} />
    </div>
  </header>;
}
