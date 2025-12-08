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
    
    stretch_rate = 1.5
    estimated_output_length = int(len(track) * stretch_rate)
    
    output_times = np.arange(estimated_output_length) / sr
    input_times = np.zeros_like(output_times)
    
    current_input_time = 0
    time_steps = 1 / sr
    
    for i in range(len(output_times)):
        output_time = output_times[i]
    
        #calculate current speed
        current_speed = 1.0 - 0.5 * (current_input_time / 30)
    
        input_times[i] = current_input_time
        current_input_time += current_speed * time_steps

    input_samples = input_times * sr
    
    track_stretched = np.interp(input_samples, np.arange(len(track)), track)
    
    return track_stretched


def time_mapping(input_time):
    
    if input_time <= 0:
        return 0
    if input_time > 30:
        # After 30s, speed is constant at 0.5x
        # time_at_30 = 60 * np.log(60 / (60 - 30))  # ≈ 41.59s
        # return time_at_30 + (input_time - 30) / 0.5
        return time_mapping(30) + (input_time - 30) / 0.5
    
    output_time = 60 * np.log(60 / (60 - input_time))
    return output_time


def annotations_time_stretch(track):
    
    original_beat_times=track.beats.times
    beat_mask = (original_beat_times >=0) & (original_beat_times <= 30)
    beats_30 = original_beat_times[beat_mask]
    beat_positions = track.beats.positions[beat_mask]
    
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
