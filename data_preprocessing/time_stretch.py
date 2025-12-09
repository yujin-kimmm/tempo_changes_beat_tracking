import mirdata
import librosa
import numpy as np
import soundfile as sf
import os


def load_dataset(dataset_name, data_home):
    
    dataset = mirdata.initialize(dataset_name, data_home)
    track_ids = dataset.track_ids
    
    return dataset, track_ids


def load_segment(dataset, track_id, start=0, end=30):
    
    track = dataset.track(track_id)
    
    y, orig_sr = track.audio
    
    target_sr = 22050
    
    y_resamp = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    
    start_sample = int(start * target_sr)
    end_sample = int(end * target_sr)
    
    track_seg = y_resamp[start_sample:end_sample]
    
    return track_seg, target_sr, track


def track_time_stretch(track, sr):
    """
    Apply time-varying stretch to audio where speed decreases linearly from 1.0x to 0.5x
    
    Args:
        track: Original audio samples (numpy array)
        sr: Sample rate (Hz)
    
    Returns:
        track_stretched: Time-stretched audio samples
    """
    
    # Set stretch rate to 1.5x (30 seconds will become ~45 seconds)
    stretch_rate = 1.5
    
    # Calculate expected output length (original length × 1.5)
    estimated_output_length = int(len(track) * stretch_rate)
    
    # Create time array for stretched audio (each sample's time in seconds)
    output_times = np.arange(estimated_output_length) / sr
    
    # Initialize array to store original audio timestamps for each output sample
    input_times = np.zeros_like(output_times)
    
    # Track current position in original audio (starts at 0 seconds)
    current_input_time = 0 
    
    # Time duration of one sample (1/sr seconds)
    time_steps = 1 / sr
    
    # For each sample in the stretched output
    for i in range(len(output_times)): 
        output_time = output_times[i]  # Current output time (not used)
    
        # Calculate playback speed at current position
        # Speed decreases linearly: 1.0 at 0s → 0.5 at 30s
        current_speed = 1.0 - 0.5 * (current_input_time / 30) 
    
        # Record which original time this output sample comes from
        input_times[i] = current_input_time
        
        # Advance in original audio based on current speed
        # Slower speed = smaller advancement = stretching effect
        current_input_time += current_speed * time_steps 

    # Convert time (seconds) to sample indices in original audio
    input_samples = input_times * sr
    
    # Interpolate to get stretched audio values from original audio
    # Maps each output sample to its corresponding position in original audio
    track_stretched = np.interp(input_samples, np.arange(len(track)), track)
    
    return track_stretched


def time_mapping(input_time):
    """
    Map original audio time to stretched audio time using logarithmic formula.
    This corresponds to the time-varying speed change (1.0x → 0.5x).
    
    Args:
        input_time: Time in original audio (seconds)
    
    Returns:
        output_time: Corresponding time in stretched audio (seconds)
    """
    
    # If time is 0 or negative, return 0
    if input_time <= 0:
        return 0
    
    # If time exceeds 30s, handle separately (speed is constant at 0.5x after 30s)
    if input_time > 30:
        # After 30s, speed is constant at 0.5x
        # time_at_30 = 60 * np.log(60 / (60 - 30))  # ≈ 41.59s
        # return time_at_30 + (input_time - 30) / 0.5
        
        # Recursively get time at 30s, then add remaining time at 0.5x speed
        return time_mapping(30) + (input_time - 30) / 0.5
    
    # Mathematical formula for time mapping with linearly decreasing speed
    # Integrates speed function: v(t) = 1.0 - 0.5*(t/30)
    output_time = 60 * np.log(60 / (60 - input_time))
    return output_time


def annotations_time_stretch(track):
    """
    Apply the same time stretch transformation to beat annotations.
    Maps original beat times to their new positions in stretched audio.
    
    Args:
        track: Track object containing beat annotations
    
    Returns:
        stretched_beat_times: Beat times adjusted for stretched audio (numpy array)
    """
    
    # Get original beat timestamps from track annotations
    original_beat_times = track.beats.times
    
    # Create boolean mask to select only beats within 0-30 second range
    beat_mask = (original_beat_times >= 0) & (original_beat_times <= 30)
    
    # Filter beats to only include those within 30 seconds
    beats_30 = original_beat_times[beat_mask]
    
    # Get corresponding beat positions (not used, but extracted for consistency)
    beat_positions = track.beats.positions[beat_mask]
    
    # Apply time_mapping function to each beat time to get stretched positions
    # Each original beat time is mapped to its new position in stretched audio
    stretched_beat_times = np.array([time_mapping(t) for t in beats_30])
    
    return stretched_beat_times


def data_time_stretch(dataset, track_id):

    track_seg, sr, track = load_segment(dataset, track_id)
    track_stretched = track_time_stretch(track_seg, sr)
    annotation_stretched = annotations_time_stretch(track)
    
    return sr, track_stretched, annotation_stretched
    

# main
def time_stretch_dataset(dataset_name, data_home):
    
    dataset, track_ids = load_dataset(dataset_name, data_home)
    
    os.makedirs(f"dataset/linear_stretch/{dataset_name}/audio", exist_ok=True)
    os.makedirs(f"dataset/linear_stretch/{dataset_name}/annotations", exist_ok=True)
    
    for track_id in track_ids:
        
        bpm = dataset.track(track_id).tempo
        
        sr, track_stretched, annotation_stretched = data_time_stretch(dataset, track_id)
    
        output_path = os.path.join(f"dataset/linear_stretch/{dataset_name}/audio", f"{track_id}_bpm{bpm}_0.5x.wav")
        sf.write(output_path, track_stretched, samplerate=sr)
        
        annotation_path = os.path.join(f"dataset/linear_stretch/{dataset_name}/annotations", f"{track_id}_bpm{bpm}_0.5x_beats.npy")
        np.save(annotation_path, annotation_stretched)
        
    print(f"Time stretched dataset created at dataset/linear_stretch/{dataset_name} directory")
