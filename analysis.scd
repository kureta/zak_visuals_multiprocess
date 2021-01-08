(
"SC_JACK_DEFAULT_INPUTS".setenv("Bitwig Studio:chords,Bitwig Studio:bass,Bitwig Studio:drums,Bitwig Studio:lead");
s = Server.local;
s.quit;
s.freeAll;
// s.options.memSize_(2.pow(20));
s.options.numInputBusChannels = 4;
s.boot;

s.waitForBoot({
	// =========== BEGIN HERE =======================
	// OSC Server address and port
	~zak = NetAddr("127.0.0.1", 8000);

	// Analyzer definitions
	// ======================================================================
	SynthDef(\chordAnalysis, { |in=0, rate=30|
		var trig, input, fft, chroma, amp, dissonance, harmonic, percussive;

		trig = Impulse.kr(rate);
		input = SoundIn.ar(in);
		fft = FFT(LocalBuf(2048), input);
		# harmonic, percussive = MedianSeparation(fft,FFTTrigger(LocalBuf(2048)),FFTTrigger(LocalBuf(2048)),2048,9);

		chroma=Lag.kr(
			Chromagram.kr(harmonic, perframenormalize: 1)
		);
		amp = Lag.kr(
			Loudness.kr(fft)
		);
		dissonance = Lag.kr(
			SensoryDissonance.kr(fft)
		);

		SendReply.kr(trig, '/chords/amp', amp);
		SendReply.kr(trig, '/chords/chroma', chroma);
		SendReply.kr(trig, '/chords/dissonance', dissonance);
	}).add;

	SynthDef(\bassAnalysis, { |in=1, rate=30|
		var trig, input, fft, amp, pitch, hasPitch, harmonic, percussive;

		trig = Impulse.kr(rate);
		input = SoundIn.ar(in);
		fft = FFT(LocalBuf(2048), input);
		// # harmonic, percussive = MedianSeparation(fft,FFTTrigger(LocalBuf(2048)),FFTTrigger(LocalBuf(2048)),2048,9);

		amp = Lag.kr(
			Loudness.kr(fft)
		);
		# pitch, hasPitch = Tartini.kr(input);
		pitch = Lag.kr(pitch.cpsmidi);
		hasPitch = EnvGen.kr(
			Env.perc(0.066, 0.5, 1.0),
			hasPitch
		);

		SendReply.kr(trig, '/bass/amp', amp);
		SendReply.kr(trig, '/bass/pitch', pitch);
		SendReply.kr(trig, '/bass/has_pitch', hasPitch);
	}).add;

	SynthDef(\drumsAnalysis, { |in=2, rate=30|
		var trig, input, fft, amp, onset, centroid;

		trig = Impulse.kr(rate);
		input = SoundIn.ar(in);
		fft = FFT(LocalBuf(2048), input);

		amp = Lag.kr(
			Loudness.kr(fft)
		);
		onset = EnvGen.kr(
			Env.perc(0.066, 0.5, 1.0),
			Onsets.kr(fft)
		);
		centroid = Lag.kr(
			SpecCentroid.kr(fft)
		);

		SendReply.kr(trig, '/drums/amp', amp);
		SendReply.kr(trig, '/drums/onset', onset);
		SendReply.kr(trig, '/drums/centroid', centroid);
	}).add;

	s.sync;

	// Analyzer instances
	// ======================================================================
	~chordAnalyzer = Synth(\chordAnalysis);
	~bassAnalyzer = Synth(\bassAnalysis);
	~drumsAnalyzer = Synth(\drumsAnalysis);

	// OSC message dispatchers
	// ======================================================================
	// -------------------------- Chords ------------------------------------
	OSCdef(\on_chords_amp, { |msg|
		~zak.sendMsg("/chords/amp", msg[3]);
	}, '/chords/amp');

	OSCdef(\on_chords_chroma, { |msg|
		var chroma = msg.copyToEnd(3);
		chroma.do {
			arg item, idx;
			~zak.sendMsg("/chords/chroma" ++ (idx), item);
		};
	}, '/chords/chroma');

	OSCdef(\on_chords_dissonance, { |msg|
		~zak.sendMsg("/chords/dissonance", msg[3]);
	}, '/chords/dissonance');
	// -------------------------- Bass --------------------------------------
	OSCdef(\on_bass_amp, { |msg|
		~zak.sendMsg("/bass/amp", msg[3]);
	}, '/bass/amp');

	OSCdef(\on_bass_has_pitch, { |msg|
		~zak.sendMsg("/bass/has_pitch", msg[3]);
	}, '/bass/has_pitch');

	OSCdef(\on_bass_pitch, { |msg|
		~zak.sendMsg("/bass/pitch", msg[3]);
	}, '/bass/pitch');
	// -------------------------- Drums -------------------------------------
	OSCdef(\on_drums_amp, { |msg|
		~zak.sendMsg("/drums/amp", msg[3]);
	}, '/drums/amp');

	OSCdef(\on_drums_onset, { |msg|
		~zak.sendMsg("/drums/onset", msg[3]);
	}, '/drums/onset');

	OSCdef(\on_drums_centroid, { |msg|
		~zak.sendMsg("/drums/centroid", msg[3]);
	}, '/drums/centroid');
};)
)
/*
Analysis UGens to check out:
Pitch
Tartini
MFCC
KeyTrack
BeatTrack2
Chromagram
Onsets
SensoryDissonance
SpecCentroid
SpecFlatness
ZeroCrossings
*/