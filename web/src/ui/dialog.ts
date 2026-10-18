// In-app confirmation and text-entry dialogs (instead of the browser's confirm()/prompt()).
// A native <dialog> opened with showModal(): focus trap, Escape to cancel, inert background.

interface DialogOptions {
  title: string;
  message?: string;
  confirm: string;
  /** Destructive action: the confirm button is styled as a warning. */
  danger?: boolean;
  /** Text entry: initial value of the field (its presence turns on the field). */
  value?: string;
  label?: string;
}

function open(o: DialogOptions): Promise<string | null> {
  return new Promise((resolve) => {
    const d = document.createElement('dialog');
    d.className = 'modal';
    const form = document.createElement('form');
    form.method = 'dialog';
    const h = document.createElement('h2');
    h.textContent = o.title;
    h.id = `modal-${crypto.randomUUID()}`;
    d.setAttribute('aria-labelledby', h.id);
    form.append(h);
    if (o.message) {
      const p = document.createElement('p');
      p.textContent = o.message;
      form.append(p);
    }
    let input: HTMLInputElement | null = null;
    if (o.value !== undefined) {
      input = document.createElement('input');
      input.type = 'text';
      input.value = o.value;
      input.maxLength = 80;
      input.setAttribute('aria-label', o.label ?? o.title);
      form.append(input);
    }
    const row = document.createElement('div');
    row.className = 'modal-actions';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'ghost';
    cancel.textContent = 'Cancel';
    cancel.onclick = () => d.close('');
    const ok = document.createElement('button');
    ok.type = 'submit';
    ok.value = 'ok';
    ok.className = o.danger ? 'primary danger' : 'primary';
    ok.textContent = o.confirm;
    row.append(cancel, ok);
    form.append(row);
    d.append(form);
    // A tap on the backdrop (outside the form) cancels.
    d.addEventListener('click', (e) => { if (e.target === d) d.close(''); });
    d.addEventListener('close', () => {
      const result = d.returnValue === 'ok' ? (input ? input.value : '') : null;
      d.remove();
      resolve(result);
    });
    document.body.append(d);
    d.showModal();
    if (input) input.select();
    else (o.danger ? cancel : ok).focus();
  });
}

/** Resolves true when the user confirms. */
export async function confirmDialog(o: Omit<DialogOptions, 'value' | 'label'>): Promise<boolean> {
  return (await open(o)) !== null;
}

/** Resolves the entered text, or null when cancelled. */
export function promptDialog(o: DialogOptions & { value: string }): Promise<string | null> {
  return open(o);
}
