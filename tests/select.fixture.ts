import { fireEvent, screen, within } from '@testing-library/react';
export function chooseOption(control: HTMLElement, value: string) {
  if (control.getAttribute('aria-expanded') !== 'true') fireEvent.click(control);
  const option = within(screen.getByRole('listbox')).getAllByRole('option').find(item => item.getAttribute('data-value') === value);
  if (!option) throw new Error('Requested test option unavailable');
  fireEvent.click(option);
}
