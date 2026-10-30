import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../../App';

afterEach(cleanup);

describe('Footer info dialogs', () => {
  it('opens a labelled dialog from the footer and closes it with Escape', async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.queryByRole('dialog')).toBeNull();
    await user.click(screen.getByRole('button', { name: 'About' }));

    const dialog = screen.getByRole('dialog', { name: 'About' });
    expect(dialog.textContent).toContain('Assistant Professor Dr. Sirawich Vachmanus');

    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('opens How to Use and License, and closes via the close button', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: 'How to Use' }));
    expect(screen.getByRole('dialog', { name: 'How to Use' })).not.toBeNull();
    await user.click(screen.getByRole('button', { name: 'Close dialog' }));
    expect(screen.queryByRole('dialog')).toBeNull();

    await user.click(screen.getByRole('button', { name: 'License' }));
    expect(screen.getByRole('dialog', { name: 'License' }).textContent).toContain('MIT License');
  });
});
