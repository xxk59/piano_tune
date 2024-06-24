# PyPianoTune

A Python module to convert music notes into piano tune

## Installation

```
pip install PyPianoTune
```

## Import module

```python
from pypianotune import PyPianoTune
```

## Sample

```python
from pypianotune import PyPianoTune
from IPython.display import Audio

# [Mandatory] Musical notes
# Full notes list can be retrieved by getPianoNotes()
# e.g. C major notes
music_notes = ['C4','D4','E4','F4','G4','A4','B4']

# [Optional] Note duration values
# Need to be aligned with count of notes
note_values = [.5, .5, .5, .5, .5, .5, .5]

pianoTuner = PyPianoTune()

Audio(pianoTuner.get_song_data(music_notes, note_values), rate=44100, autoplay=True)
```

## License

MIT License
