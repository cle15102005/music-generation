import pretty_midi
import soundfile as sf

# Paths
midi_path = r"maestro-v3.0.0-midi\maestro-v3.0.0\2004\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"  # Your MIDI file
sf2_path = "GeneralUser-GS.sf2"  # Path to your .sf2 SoundFont file
output_wav_path = "output.wav"  # Output audio file

# Load MIDI
midi_data = pretty_midi.PrettyMIDI(midi_path)

# Synthesize audio using SoundFont
audio_data = midi_data.fluidsynth(sf2_path=sf2_path)

# Save to WAV
sf.write(output_wav_path, audio_data, 44100)

print("✅ Conversion complete: output.wav")
