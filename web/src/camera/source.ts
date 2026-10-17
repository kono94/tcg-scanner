// Camera sources. The video-file mock feeds a MediaStream through the same path as getUserMedia.

export interface CameraHandle {
  stream: MediaStream;
  label: string;
  /** True for the video-file mock; the UI labels it so results are never mistaken for camera tests. */
  mock: boolean;
  stop(): void;
  /** Pause frame delivery without closing the camera (instant resume, no new permission prompt). */
  setActive(active: boolean): void;
}

export async function openCamera(): Promise<CameraHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(window.isSecureContext ? 'This browser has no camera access' : 'Camera access needs HTTPS (or localhost)');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    // 1440p; only the visible centre is captured (cropped), and capped at 2048 px on iOS (tab memory limit).
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 2560 }, height: { ideal: 1440 } },
  });
  const track = stream.getVideoTracks()[0];
  const s = track.getSettings();
  return {
    stream,
    label: `${track.label || 'Camera'} · ${s.width ?? '?'}×${s.height ?? '?'}`,
    mock: false,
    stop: () => stream.getTracks().forEach((t) => t.stop()),
    setActive: (active) => stream.getVideoTracks().forEach((t) => { t.enabled = active; }),
  };
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
