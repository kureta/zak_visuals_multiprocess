(
s.freeAll;
s.quit;
// Analyzer definitions
// ======================================================================

Spec.add(\rate, [1,60]);
Spec.add(\scale, [-3, 3]);
Spec.add(\lag, [0, 1.0]);


~chordsAnalDef = SynthDef(\chordsAnalysis, { |in=0|
	var trig, input, fft, chroma, amp, dissonance, harmonic, percussive, lag, cc, other_trig, mfcc, mfcc_trig;

	trig = Impulse.kr(\rate.kr(60));
	input = SoundIn.ar(in);
	fft = FFT(LocalBuf(2048), input, hop: 0.5, wintype: 1);
	mfcc = MFCC.kr(fft, nunmcoeff: 64);
	mfcc_trig = Changed.kr(mfcc);
	# harmonic, percussive = MedianSeparation(fft,FFTTrigger(LocalBuf(2048)),FFTTrigger(LocalBuf(2048)),2048,9);
	lag = \lag.kr(0.05);

	cc = Chromagram.kr(harmonic);
	other_trig = Changed.kr(cc);
	chroma=Lag3.kr(
		cc.sqrt * 10.pow(\scale.kr(2.23)),
		lag
	);
	amp = Lag3.kr(
		RMS.kr(input),
		lag
	);
	dissonance = Lag.kr(
		SensoryDissonance.kr(fft)
	);

	SendReply.kr(trig, '/chords/amp', amp);
	SendReply.kr(other_trig, '/chords/chroma', chroma);
	SendReply.kr(trig, '/chords/dissonance', dissonance);
	SendReply.kr(mfcc_trig, '/chords/mfcc', mfcc);
});

~bassAnalDef = SynthDef(\bassAnalysis, { |in=1|
	var trig, input, fft, amp, pitch, hasPitch, harmonic, percussive;

	trig = Impulse.kr(\rate.kr(40));
	input = SoundIn.ar(in);
	// fft = FFT(LocalBuf(2048), input);
	// # harmonic, percussive = MedianSeparation(fft,FFTTrigger(LocalBuf(2048)),FFTTrigger(LocalBuf(2048)),2048,9);

	amp = Lag.kr(
		Clip.kr((10 * RMS.kr(input).squared.log10 + 40) / 40)
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
});

~drumsAnalDef = SynthDef(\drumsAnalysis, { |in=2|
	var trig, input, fft, amp, onset, centroid, release, onset_;

	trig = Impulse.kr(\rate.kr(40));
	input = SoundIn.ar(in);
	fft = FFT(LocalBuf(2048), input);
	release = 0.5;

	amp = Lag.kr(
		RMS.kr(input),
		1/30
	);
	onset_ = Onsets.kr(fft, threshold: 0.3, odftype: 'magsum', mingap: 5);
	onset = EnvGen.kr(
		Env.perc(0, 3/40, 1.0, -2),
		onset_
		);

	centroid = Lag.kr(
		Clip.kr(SpecCentroid.kr(fft), 20, 24000).cpsmidi / 127,
		release
	);

	SendReply.kr(trig, '/drums/amp', amp);
	SendReply.kr(trig, '/drums/onset', onset);
	SendReply.kr(trig, '/drums/centroid', centroid);
});

// OSC message dispatchers
// ======================================================================

// OSC Server address and port
~zak = NetAddr("127.0.0.1", 8000);
// -------------------------- Chords ------------------------------------
OSCdef(\on_chords_amp, { |msg|
	~zak.sendMsg("/chords/amp", msg[3]);
}, '/chords/amp');

OSCdef(\on_chords_chroma, { |msg|
	var chords = msg.copyToEnd(3);
	~zak.sendMsg("/chords/chroma", *chords);
}, '/chords/chroma');

OSCdef(\on_chords_dissonance, { |msg|
	~zak.sendMsg("/chords/dissonance", msg[3]);
}, '/chords/dissonance');

OSCdef(\on_chords_mfcc, { |msg|
	var mfcc = msg.copyToEnd(3);
	~zak.sendMsg("/chords/mfcc", *mfcc);
}, '/chords/mfcc');

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
	// msg[3].postln;
}, '/drums/onset');

OSCdef(\on_drums_centroid, { |msg|
	~zak.sendMsg("/drums/centroid", msg[3]);
}, '/drums/centroid');

// Server communication
// ======================================================================
// This weird code organization is for understanding the client/server architecture of supercollider better.
// Only code below this point communicates with the server.
"SC_JACK_DEFAULT_INPUTS".setenv("Bitwig Studio:chords,Bitwig Studio:bass,Bitwig Studio:drums,Bitwig Studio:lead");
s.options.numInputBusChannels = 4;
s.boot;
s.waitForBoot{
	// Add SynthDefs to the server
	~chordsAnalDef.add;
	~bassAnalDef.add;
	~drumsAnalDef.add;
	s.sync;

	// Analyzer instances
	// SynthDescLib.global[\chordsAnalysis].makeWindow;
	// SynthDescLib.global[\bassAnalysis].makeWindow;
	// SynthDescLib.global[\drumsAnalysis].makeWindow;

	~chordsAnalyzer = Synth(\chordsAnalysis);
	~bassAnalyzer = Synth(\bassAnalysis);
	~drumsAnalyzer = Synth(\drumsAnalysis);
};
)
