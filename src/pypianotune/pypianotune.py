import numpy as np

C4_FACTOR = [0.73168812, 0.16083298, 0.06460201, 0.00744849, 0.0174458, 0.00318635, 0.01479625]
RIGHT_HAND_FACTOR = [0.68, 0.26, 0.03, 0., 0.03]
LEFT_HAND_FACTOR = [0.73, 0.16, 0.06, 0.01, 0.02, 0.01, 0.01]
RIGHT_HAND_LENGTH = [0.01, 0.6, 0.29, 0.1]
LEFT_HAND_LENGTH = [0.01, 0.29, 0.6, 0.1]
DECAY = [0.05, 0.02, 0.005, 0.1]
SUSTAIN_LEVEL = 0.1
A4_FREQUENCY = 440  #Frequency of Note A4
SAMPLE_RATE = 44100
AMPLITUDE = 4096
DUPLE = 2
TRIPLE = 3
QUADRUPLE = 4

DEMO_RIGHT_HAND_NOTES = ['','A4','','A4',
                        '','A4','','A4',
                        'E4','A4','A4','B4',
                        'C5','C5','A4','A4','A4','C5',
                        'B4','G4','G4','G4',
                        'B4','B4','A4','A4','A4',
                        'E4','A4','A4','B4',
                        'C5','C5','A4','A4','A4','C5',
                        'E5','E5','E5','D5','C5','C5','B4','B4',
                        'C5','A4','A4','A4','A4','C5',
                        'E5','E5','D5','C5',
                        'B4','G4','G4','G4','G4',
                        'D5','D5','D5','D5','C5','B4',
                        'C5','A4','A4','A4','A4','C5',
                        'E5','E5','D5','C5',
                        'B4','G4','G4','G4','G4',
                        'D5','D5','D5','D5','C5','B4',
                        'C5','A4','A4','A4']

DEMO_RIGHT_HAND_NOTE_VALUES = [.25,.25,.25,.25,
                            .25,.25,.25,.25,
                            .25,.25,.375,.125,
                            .125,.125,.25,.25,.125,.125,
                            .25,.25,.25,.25,
                            .125,.125,.25,.375,.125,
                            .25,.25,.375,.125,
                            .125,.125,.25,.25,.125,.125,
                            .125,.125,.125,.125,.125,.125,.125,.125,
                            .25,.125,.125,.25,.125,.125,
                            .25,.25,.25,.25,
                            .25,.25,.25,.125,.125,
                            .125,.125,.125,.125,.25,.25,
                            .25,.125,.125,.25,.125,.125,
                            .25,.25,.25,.25,
                            .25,.25,.25,.125,.125,
                            .125,.125,.125,.125,.25,.25,
                            .25,.25,.375,.125]

DEMO_LEFT_HAND_NOTES = ['A3','','E3','',
                        'A3','','E3','',
                        'A3','E4','A3','E4',
                        'A3','E4','A3','E4',
                        'E3','D4','E3','D4',
                        'A3','E4','A3','E4',
                        'A3','E4','A3','E4',
                        'A3','E4','A3','E4',
                        'E3','D4','E3','D4',
                        'A3','E4','A3','E4',
                        'A3','E4','A3','E4',
                        'E3','D4','E3','D4',
                        'E3','D4','E3','D4',
                        'A3','E4','A3','E4',
                        'A3','E4','A3','E4',
                        'E3','D4','E3','D4',
                        'E3','D4','E3','D4',
                        'A3','E4','A3','E4']

DEMO_LEFT_HAND_NOTE_VALUES = [.25] * 72

class PyPianoTune:

    def __init__(self):
        # Define default values for various parameters
        self.base_freq = A4_FREQUENCY
        self.bar_value = DUPLE
        self.factor = C4_FACTOR
        self.length = RIGHT_HAND_LENGTH
        self.decay = DECAY
        self.sustain_level = SUSTAIN_LEVEL
        self.sample_rate = SAMPLE_RATE
        self.amplitude = AMPLITUDE

    # Get the fundamental frequencies of the 88 keys of the piano, as well as the stop
    def get_piano_notes(self):
        # White keys are in Uppercase and black keys (sharps) are in lowercase
        octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
        keys = np.array([x+str(y) for y in range(0,9) for x in octave])
        # Trim to standard 88 keys
        start = np.where(keys == 'A0')[0][0]
        end = np.where(keys == 'C8')[0][0]
        keys = keys[start:end+1]
        
        note_freqs = dict(zip(keys, [2**((n+1-49)/12)*self.base_freq for n in range(len(keys))]))
        note_freqs[''] = 0.0 # stop
        return note_freqs

    # Generate sound wave corresponding to frequency
    def get_sine_wave(self, frequency, duration, sample_rate=None, amplitude=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if amplitude is None:
            amplitude = self.amplitude
        
        t = np.linspace(0, duration, int(sample_rate*duration)) # Time axis
        wave = amplitude*np.sin(2*np.pi*frequency*t)
        return wave

    # Apply overtones to the fundamental tone
    def apply_over_tones(self, frequency, duration, factor, sample_rate=None, amplitude=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if amplitude is None:
            amplitude = self.amplitude

        assert abs(1-sum(factor)) < 1e-8
        
        frequencies = np.minimum(np.array([frequency*(x+1) for x in range(len(factor))]), sample_rate//2)
        amplitudes = np.array([amplitude*x for x in factor])
        
        fundamental = self.get_sine_wave(frequencies[0], duration, sample_rate, amplitudes[0])
        for i in range(1, len(factor)):
            overtone = self.get_sine_wave(frequencies[i], duration, sample_rate, amplitudes[i])
            fundamental += overtone
        return fundamental

    # Apply ADSR effects
    def get_adsr_weights(self, frequency, duration, length, decay, sustain_level, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate

        assert abs(sum(length)-1) < 1e-8
        assert len(length) == len(decay) == 4
        
        intervals = int(duration*frequency)
        len_A = np.maximum(int(intervals*length[0]),1)
        len_D = np.maximum(int(intervals*length[1]),1)
        len_S = np.maximum(int(intervals*length[2]),1)
        len_R = np.maximum(int(intervals*length[3]),1)
        
        decay_A = decay[0]
        decay_D = decay[1]
        decay_S = decay[2]
        decay_R = decay[3]
        
        A = 1/np.array([(1-decay_A)**n for n in range(len_A)])
        A = A/np.nanmax(A)
        D = np.array([(1-decay_D)**n for n in range(len_D)])
        D = D*(1-sustain_level)+sustain_level
        S = np.array([(1-decay_S)**n for n in range(len_S)])
        S = S*sustain_level
        R = np.array([(1-decay_R)**n for n in range(len_R)])
        R = R*S[-1]
        
        weights = np.concatenate((A,D,S,R))
        smoothing = np.array([0.1*(1-0.1)**n for n in range(5)])
        smoothing = smoothing/np.nansum(smoothing)
        weights = np.convolve(weights, smoothing, mode='same')
        
        if intervals == 0:
            return 0
        weights = np.repeat(weights, int(sample_rate*duration/intervals))
        tail = int(sample_rate*duration-weights.shape[0])
        if tail > 0:
            weights = np.concatenate((weights, weights[-1]-weights[-1]/tail*np.arange(tail)))
        return weights

    # Apply pedal effects to notes
    def apply_pedal(self, note_values, bar_value):
        new_values = []

        # Check that we have whole number of bars
        assert sum(note_values) % bar_value == 0

        start = 0
        while True:
            # Count total duration from end of last bar
            cum_value = np.cumsum(np.array(note_values[start:]))
            # Find end of this 
            end = np.where(cum_value == bar_value)[0][0]
            if end == 0: # If the note takes up the whole bar
                new_values += [note_values[start]]
            else:
                this_bar = np.array(note_values[start:start+end+1])
                # New value of note is the remainder of bar = (total duration of bar) - (cumulative duration thus far)
                new_values += [bar_value-np.sum(this_bar[:i]) for i in range(len(this_bar))]
            start += end+1
            if start == len(note_values):
                break
        return new_values

    # Generate song data according to the music notes and duration data
    def get_song_data(self,
                    music_notes, 
                    note_values=None, 
                    bar_value=None, 
                    factor=None, 
                    length=None, 
                    decay=None, 
                    sustain_level=None, 
                    sample_rate=None, 
                    amplitude=None):

        # Padding of the note values
        base_note_value=.5
        if note_values is None:
            note_values = [base_note_value] * len(music_notes)

        if bar_value is None:
            bar_value = self.bar_value
        if factor is None:
            factor = self.factor 
        if length is None:
            length = self.length
        if decay is None:
            decay = self.decay
        if sustain_level is None:
            sustain_level = self.sustain_level
        if sample_rate is None:
            sample_rate = self.sample_rate
        if amplitude is None:
            amplitude = self.amplitude

        # Get note frequencies
        note_freqs = self.get_piano_notes()

        # Padding and align the music notes with the bar_value
        residual = np.sum(note_values) % bar_value
        if residual > 0:
            music_notes += ['']
            note_values += [bar_value - residual]

        frequencies = [note_freqs[note.strip()] for note in music_notes]

        # Get new note durations with sustain applied
        new_values = self.apply_pedal(note_values, bar_value)
        # End of each note without sustain
        end_idx = np.cumsum(np.array(note_values)*sample_rate).astype(int)
        # Start of each note
        start_idx = np.concatenate(([0], end_idx[:-1]))
        # End of note with sustain
        end_idx = np.array([start_idx[i]+new_values[i]*sample_rate for i in range(len(new_values))]).astype(int)
        
        # Total duration of the piece
        duration = int(sum(note_values)*sample_rate)    
        song = np.zeros((duration,))
        for i in range(len(music_notes)):
            # Fundamental + overtones
            if music_notes[i] != '':
                this_note = self.apply_over_tones(frequencies[i], new_values[i], factor)
                # ADSR model
                weights = self.get_adsr_weights(frequencies[i], new_values[i], length, 
                                        decay, sustain_level)
                song[start_idx[i]:end_idx[i]] += this_note*weights

        song = song*(amplitude/np.max(song))
        return song

    # Combine one or more song data for Audio readiness
    def append_song_data(self, *args):
        # Check if all arguments are ndarrays
        if not all(isinstance(arr, np.ndarray) for arr in args):
            raise ValueError("All arguments must be NumPy ndarrays")

        max_length = max(arr.shape[0] for arr in args)
        combined_array = np.zeros(max_length, dtype=args[0].dtype)
        for arr in args:
            if arr.shape[0] < max_length:
                combined_array += np.pad(arr, (0, max_length-arr.shape[0]), mode='constant', constant_values=0)
            else:
                combined_array += arr

        data = combined_array * (self.amplitude/np.max(combined_array))
        return data