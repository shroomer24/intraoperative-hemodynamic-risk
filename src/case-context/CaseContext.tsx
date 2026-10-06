import { caseLabel } from '../data/types';
import type { CaseOrdinal } from '../data/types';
import { caseContexts } from './records';
import type { CaseContextRecord } from './records';

function available(value: string | undefined): string | undefined {
  const text = value?.trim();
  return text && text !== 'Not available' ? text : undefined;
}
export function contextLine(record: CaseContextRecord | undefined): string {
  const procedure = available(record?.procedure);
  const department = available(record?.department);
  if (!procedure) return department ?? 'Surgical context not available';
  return [procedure, department, available(record?.approach)].filter(Boolean).join(' · ');
}
export function CaseContext({ ordinal }: { ordinal: CaseOrdinal }) {
  // Resolve directly from the current public ordinal on every render: no async/cache state.
  const record = caseContexts.find(item => item.case_ordinal === ordinal);
  return <p className="case-context" data-case-ordinal={ordinal}
    aria-label={`Surgical context for ${caseLabel(ordinal)}`}>{contextLine(record)}</p>;
}
