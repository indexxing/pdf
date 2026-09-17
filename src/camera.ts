export async function openCamera(video: HTMLVideoElement): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      window.isSecureContext
        ? 'This browser does not support camera access.'
        : 'Camera access requires HTTPS (or localhost).',
    );
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 3840 },
      height: { ideal: 2160 },
    },
  });

  // Continuous autofocus where supported (mostly Android Chrome); ignored elsewhere.
  const [track] = stream.getVideoTracks();
  const capabilities = track.getCapabilities?.() as MediaTrackCapabilities & { focusMode?: string[] };
  if (capabilities?.focusMode?.includes('continuous')) {
    track.applyConstraints({ advanced: [{ focusMode: 'continuous' } as MediaTrackConstraintSet] }).catch(() => {});
  }

  // Mirror front-facing cameras (laptop webcams usually don't report a facing mode) so moving
  // the page left moves it left on screen, like a video call. Scans themselves are never mirrored.
  const facing = track.getSettings().facingMode;
  const mirrored = facing ? facing === 'user' : !window.matchMedia('(pointer: coarse)').matches;
  video.closest('.stage')?.classList.toggle('mirrored', mirrored);

  video.srcObject = stream;
  await video.play();
  if (!video.videoWidth) {
    await new Promise((resolve) => video.addEventListener('loadedmetadata', resolve, { once: true }));
  }
  return stream;
}

export function describeCameraError(err: unknown): string {
  if (err instanceof DOMException) {
    if (err.name === 'NotAllowedError') return 'Camera permission was denied. Allow access in your browser settings and try again.';
    if (err.name === 'NotFoundError') return 'No camera was found on this device.';
    if (err.name === 'NotReadableError') return 'The camera is in use by another app.';
  }
  return err instanceof Error ? err.message : 'Could not start the camera.';
}
