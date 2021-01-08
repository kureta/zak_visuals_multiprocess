(
"SC_JACK_DEFAULT_INPUTS".setenv("Bitwig Studio:Speakers-L,Bitwig Studio:Speakers-R");
s = Server.local;
s.quit;
s.options.memSize_(2.pow(20));
s.options.numInputBusChannels = 4;
s.boot;
)

// ==============================================
// =========== BEGIN HERE =======================
// ==============================================

(
// SynthDef to analyse audio signal for amplitude
(
SynthDef(\amplitudeAnalysis, { |in=0, rate=30|
	var trig, input, fft, chroma, amp, onset, pitch, hasPitch, dissonance, harmonic, percussive;

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
	onset = Onsets.kr(fft);
	# pitch, hasPitch = Tartini.kr(input);
	pitch = Lag.kr(pitch.cpsmidi);
	dissonance = Lag.kr(
		SensoryDissonance.kr(fft)
	);

	SendReply.kr(trig, '/amp', amp);
	SendReply.kr(trig, '/chroma', chroma);
	SendReply.kr(trig, '/onset', onset);
	SendReply.kr(trig, '/has_pitch', hasPitch);
	SendReply.kr(trig, '/pitch', pitch);
	SendReply.kr(trig, '/dissonance', dissonance);
}).add;
);

// an instance of the Synth
~mySynth = Synth(\amplitudeAnalysis);
// OSC Server address and port
~zaxNetAddr = NetAddr("127.0.0.1", 8000);
// Chroma range
~range = Range.new(3, 12);
// The listener for the analysis data
// Receives messages from the audio server, sends them to OSC Server
(
OSCdef(\listen_amp, { |msg|
	~zaxNetAddr.sendMsg("/controls/amp", msg[3]);
}, '/amp');

OSCdef(\listen_chroma, { |msg|
	var chroma = msg.copyToEnd(3);
	~range.do {
		arg item;
		~zaxNetAddr.sendMsg("/controls/chroma" ++ (item-3), msg[item]);
	};
}, '/chroma');
);

OSCdef(\listen_onset, { |msg|
	~zaxNetAddr.sendMsg("/controls/onset", msg[3]);
}, '/onset');

OSCdef(\listen_has_pitch, { |msg|
	~zaxNetAddr.sendMsg("/controls/has_pitch", msg[3]);
}, '/has_pitch');

OSCdef(\listen_pitch, { |msg|
	~zaxNetAddr.sendMsg("/controls/pitch", msg[3]);
}, '/pitch');

OSCdef(\listen_dissonance, { |msg|
	~zaxNetAddr.sendMsg("/controls/dissonance", msg[3]);
}, '/dissonance');

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