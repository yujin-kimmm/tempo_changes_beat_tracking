import mirdata
import shutil
import numpy as np
import os
import soundfile as sf

from data_preprocessing.time_stretch import load_dataset

def load_dataset(dataset_name, data_home):
    
    dataset = mirdata.initialize(dataset_name, data_home)
    track_ids = dataset.track_ids
    
    return dataset, track_ids

rwc_popular = "rwc_popular"
rwc_home = "/Users/yujinkim/Google Drive/My Drive/Course/FA25/MIR/Final/rwc_popular"

def baseline(dataset_name, data_home, output_dir="./dataset", segment_start=0, segment_end=30):
    
    audio_dir = os.path.join(output_dir, "baseline/audio")
    annot_dir = os.path.join(output_dir, "baseline/annotations")
    
    # Clean up existing directories to ensure overwrite
    if os.path.exists(audio_dir):
        shutil.rmtree(audio_dir)
    if os.path.exists(annot_dir):
        shutil.rmtree(annot_dir)
        
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)
    
    dataset, track_ids = load_dataset(dataset_name, data_home)
    
    # Calculate number of pairs
    n_files = len(track_ids)

    for i in range(n_files):
        # Select 2 unique tracks sequentially
        track_id = track_ids[i]
        
        track = dataset.track(track_id)
        
        # --- Audio Processing ---
        y, sr = track.audio
        
        start_sample = int(segment_start * sr)
        end_sample = int(segment_end * sr)
        
        # Slice audio (handling bounds safely)
        
        y_seg = y[start_sample:end_sample]


        audio = y_seg
        

        bpm = track.tempo
        
        base_filename = f"{track_id}_{bpm}bpm_segmented.wav"

        # Save audio
        out_wav_path = os.path.join(audio_dir, f"{base_filename}.wav")
        sf.write(out_wav_path, audio, sr)
        
        # --- Annotation Processing (Beats) ---
        # Track 1
        t1_times = track.beats.times
        t1_mask = (t1_times >= segment_start) & (t1_times <= segment_end)
        
        t1_times_filt = t1_times[t1_mask]
        
        # Shift Track 1 timestamps to start at 0
        t1_times_shifted = t1_times_filt - segment_start
            
        # Save annotations
        np.save(os.path.join(annot_dir, f"{base_filename}_times.npy"), t1_times_shifted)


    print("Generation complete.")
