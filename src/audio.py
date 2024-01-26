import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import time


def process_audio_chunk(chunk):
    # Convert audio chunk to mono
    mono_chunk = np.mean(chunk, axis=1)
    fft_result = np.fft.fft(mono_chunk)
    fft_magnitude = np.abs(fft_result)

    # Define sample rate and bass frequency range
    sample_rate = 44100
    bass_range = (20, 150)  # Typical bass range in Hz

    # Calculate indices for bass frequencies
    freqs = np.fft.fftfreq(len(fft_magnitude), 1 / sample_rate)
    bass_indices = np.where((freqs >= bass_range[0]) & (freqs <= bass_range[1]))[0]

    # Extract and normalize bass amplitude
    bass_amplitude = np.max(fft_magnitude[bass_indices])
    normalized_bass_amplitude = bass_amplitude / np.max(fft_magnitude)

    print(f"Normalized bass amplitude: {normalized_bass_amplitude}")

def stream_audio_file(file_path, chunk_size=1024):
    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(file_path)
    audio = audio.set_channels(1).set_frame_rate(44100)  # Mono and 44100Hz
    samples = np.array(audio.get_array_of_samples())

    sample_rate = 44100  # Samples per second
    chunk_duration = chunk_size / sample_rate  # Duration of each chunk in seconds

    # Stream and process in chunks
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        process_audio_chunk(chunk.reshape(-1, 1))

        time.sleep(chunk_duration)  # Delay for real-time playback

# Replace 'path_to_your_mp3_file.mp3' with the path to your MP3 file
stream_audio_file('baby-mandala-169039.mp3')
