- Latents will be controled by scalars.
- Every scalar will affect either magnitude or direction of a latent vector.
- There are 17 2D Noises, 18 ws, and  1 z. Controlling by z may be a different mode later on.

## Chords
chroma, loudness, dissonance
## Bass
pitch, loudness, onset
## Drums
onset, loudness, spectral centroid (kick or snare - brightness)


# TODO
- [ ] `has_pitch` and `onset` miss some beats, because we are sampling them @30Hz
- [ ] Animate direction chages. Mostly triggered by onset detection.
- [ ] Process separate channels:
  - Bass
  - Drums
  - Chords
  - Lead?
- [x] Use a fast, external program (like supercollider) for realtime feature extraction
- [x] Features to be used:
    - [x] loudness
    - [x] onset
    - [x] chromagram
    - [x] separate percussive and harmonic parts
    - [x] pitch
    - [x] dissonance
- [ ] Noises will be controlled by b&w images (later maybe via an additional `/dev/video1` + gstreamer)
