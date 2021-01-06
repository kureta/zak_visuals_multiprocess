* Latents will be controled by scalars
* Every scalar will affect either magnitude or direction of a latent vector
* Noises will be controlled by b&w images (later maybe via an additional `/dev/video1` + gstreamer)
* There are 17 2D Noises, 18 ws, and  1 z
* Use a fast, external program (like supercollider) for realtime feature extraction
* Features to be used:
    * loudness
    * onset
    * chromagram
    * separate percussive and harmonic parts