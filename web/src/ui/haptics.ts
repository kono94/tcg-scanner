// Capture confirmation for a user who looks at the binder, not the screen: a short vibration,
// or a soft click where the browser has no vibration API (iOS Safari).
let audio: AudioContext | null = null;

/** Audio must be unlocked by a user gesture on iOS: create/resume the context on the first tap. */
export function primeCaptureFeedback(): void {
  const unlock = () => {
    try {
      audio ??= new AudioContext();
      void audio.resume();
    } catch { /* no Web Audio */ }
  };
  window.addEventListener('pointerdown', unlock, { capture: true, passive: true });
}

export function captureFeedback(): void {
  if (typeof navigator.vibrate === 'function' && navigator.vibrate(35)) return;
  if (!audio || audio.state !== 'running') return;
  try {
    const t = audio.currentTime;
    const osc = audio.createOscillator(), gain = audio.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(1400, t);
    osc.frequency.exponentialRampToValueAtTime(700, t + 0.05);
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.18, t + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.07);
    osc.connect(gain).connect(audio.destination);
    osc.start(t);
    osc.stop(t + 0.08);
  } catch { /* feedback is optional */ }
}
