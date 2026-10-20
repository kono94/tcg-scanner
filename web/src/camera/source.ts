// Camera sources. The video-file mock feeds a MediaStream through the same path as getUserMedia.

export interface CameraHandle {
  /** Current stream; replaced when reopen() acquires the camera again. */
  readonly stream: MediaStream;
  label: string;
  /** True for the video-file mock; the UI labels it so results are never mistaken for camera tests. */
  mock: boolean;
  stop(): void;
  /** Pause frame delivery without closing the camera (instant resume, no new permission prompt). */
  setActive(active: boolean): void;
  /** The camera hardware is released (camera light off): reopen() before use. */
  readonly suspended: boolean;
  /** Release the camera hardware but keep this handle; reopen() acquires it again (may prompt on some browsers). */
  suspend(): void;
  reopen(): Promise<void>;
}

export async function openCamera(): Promise<CameraHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(window.isSecureContext ? 'This browser has no camera access' : 'Camera access needs HTTPS (or localhost)');
  }
  const acquire = () => navigator.mediaDevices.getUserMedia({
    audio: false,
    // 1440p; only the visible centre is captured (cropped), and capped at 2048 px on iOS (tab memory limit).
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 2560 }, height: { ideal: 1440 } },
  });
  const end = (s: MediaStream) => s.getTracks().forEach((t) => t.stop());
  let stream = await acquire();
  /** False after suspend()/stop(): a reopen still in flight closes what it gets. */
  let wanted = true;
  let opening: Promise<void> | null = null;
  const track = stream.getVideoTracks()[0];
  const s = track.getSettings();
  const handle: CameraHandle = {
    get stream() { return stream; },
    label: `${track.label || 'Camera'} · ${s.width ?? '?'}×${s.height ?? '?'}`,
    mock: false,
    // Ended also covers a camera the system took away (another app, a call).
    get suspended() { return stream.getVideoTracks().every((t) => t.readyState === 'ended'); },
    stop: () => { wanted = false; end(stream); },
    suspend: () => { wanted = false; end(stream); },
    setActive: (active) => stream.getVideoTracks().forEach((t) => { t.enabled = active; }),
    reopen: async () => {
      wanted = true;
      if (!handle.suspended) return;
      opening ??= acquire().then((next) => { if (wanted) stream = next; else end(next); }).finally(() => { opening = null; });
      await opening;
    },
  };
  return handle;
}

type CapturableVideo = HTMLVideoElement & { captureStream?: () => MediaStream; mozCaptureStream?: () => MediaStream };

/** Play a local video file and expose it as a camera stream. */
export async function openVideoMock(url: string, name: string): Promise<CameraHandle> {
  const video = document.createElement('video') as CapturableVideo;
  video.src = url;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  // Keep it in the document (hidden) so browsers keep decoding frames.
  video.style.cssText = 'position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;left:0;top:0';
  document.body.appendChild(video);
  await video.play();
  const capture = video.captureStream?.bind(video) ?? video.mozCaptureStream?.bind(video);
  if (!capture) {
    video.remove();
    throw new Error('This browser cannot turn a video into a camera stream (captureStream)');
  }
  const stream = capture();
  return {
    stream,
    label: `Test video · ${name} · ${video.videoWidth}×${video.videoHeight}`,
    mock: true,
    // No hardware to release: a paused test video stays attached.
    suspended: false,
    suspend: () => {},
    reopen: async () => {},
    setActive: (active) => {
      stream.getVideoTracks().forEach((t) => { t.enabled = active; });
      if (active) void video.play(); else video.pause();
    },
    stop: () => {
      stream.getTracks().forEach((t) => t.stop());
      video.pause();
      video.removeAttribute('src');
      video.load();
      video.remove();
    },
  };
}

export interface MockVideo {
  name: string;
  url: string;
  note: string;
}

/** Local test videos listed in public/media/manifest.json (not committed). */
export async function listMockVideos(): Promise<MockVideo[]> {
  try {
    const r = await fetch('media/manifest.json', { cache: 'no-store' });
    if (!r.ok) return [];
    const list = (await r.json()) as { videos: MockVideo[] };
    return list.videos ?? [];
  } catch {
    return [];
  }
}
