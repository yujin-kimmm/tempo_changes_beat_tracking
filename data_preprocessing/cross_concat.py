import mirdata
import shutil
import os
import numpy as np
import soundfile as sf
import librosa


def concatenate_tracks_diff_genre(rwc, rwc_home, gtzan, gtzan_home, output_dir="./dataset/cross_genre", segment_start=0, segment_end=15):
    """
    Concatenates RWC Popular tracks with GTZAN tracks for each genre.
    """
    
    rwc_dataset = mirdata.initialize(rwc, rwc_home)
    gtzan_dataset = mirdata.initialize(gtzan, gtzan_home)
    
    
    # THERE IS ALSO JAZZ IN GTZAN
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Get all RWC track IDs
    rwc_track_ids = rwc_dataset.track_ids
    
    for genre in genres:
        print(f"Processing genre: {genre}...")
        
        # Structure: output_dir/genre/genre_audio and output_dir/genre/genre_annotations
        genre_base_dir = os.path.join(output_dir, genre)
        current_genre_audio_dir = os.path.join(genre_base_dir, f"{genre}_audio")
        current_genre_annot_dir = os.path.join(genre_base_dir, f"{genre}_annotations")
        
        # Cleanup
        if os.path.exists(current_genre_audio_dir):
            shutil.rmtree(current_genre_audio_dir)
        if os.path.exists(current_genre_annot_dir):
            shutil.rmtree(current_genre_annot_dir)
            
        os.makedirs(current_genre_audio_dir, exist_ok=True)
        os.makedirs(current_genre_annot_dir, exist_ok=True)
        
        # Get GTZAN tracks for this genre
        gtzan_genre_ids = [tid for tid in gtzan_dataset.track_ids if tid.startswith(f"{genre}.")]
        gtzan_genre_ids.sort() 
        
        # Number of files is limited by whichever dataset runs out first
        n_files = min(len(rwc_track_ids), len(gtzan_genre_ids))
        
        for i in range(n_files):
            rwc_id = rwc_track_ids[i]
            gtzan_id = gtzan_genre_ids[i]
            
            try:
                track1 = rwc_dataset.track(rwc_id)
                track2 = gtzan_dataset.track(gtzan_id)
                
                # Check for annotations first
                if track1.beats is None or track2.beats is None:
                    raise ValueError("Missing beat annotations")

                # --- Audio Processing ---
                y1, sr1 = track1.audio
                y2, sr2 = track2.audio
                
                start_sample = int(segment_start * sr1)
                end_sample = int(segment_end * sr1)
                
                # Slice Track 1 (RWC)
                if len(y1) < end_sample: 
                    y1_seg = y1[start_sample:]
                else:
                    y1_seg = y1[start_sample:end_sample]
                    
                # Slice Track 2 (GTZAN)
                if len(y2) < int(segment_end * sr2):
                     s2_start = int(segment_start * sr2)
                     y2_seg = y2[s2_start:]
                else:
                     s2_start = int(segment_start * sr2)
                     s2_end = int(segment_end * sr2)
                     if len(y2) < s2_end:
                         y2_seg = y2[s2_start:]
                     else:
                         y2_seg = y2[s2_start:s2_end]

                # Resample y2 to sr1 if needed
                if sr1 != sr2:
                    y2_seg = librosa.resample(y2_seg, orig_sr=sr2, target_sr=sr1)

                # Match channels
                if y1_seg.ndim > 1 and y2_seg.ndim == 1:
                    if y1_seg.shape[0] < y1_seg.shape[-1] and y1_seg.ndim==2: 
                        y2_seg = np.stack([y2_seg, y2_seg], axis=0) 
                        concat_axis = 1
                    else:
                        y2_seg = np.stack([y2_seg, y2_seg], axis=-1)
                        concat_axis = 0
                elif y1_seg.ndim == 1 and y2_seg.ndim > 1:
                    if y2_seg.shape[0] < y2_seg.shape[-1]: 
                        y2_seg = np.mean(y2_seg, axis=0)
                        concat_axis = 0 
                    else: 
                        y2_seg = np.mean(y2_seg, axis=1)
                        concat_axis = 0
                else:
                    if y1_seg.ndim == 2 and y1_seg.shape[0] < y1_seg.shape[-1]:
                        concat_axis = 1
                    else:
                        concat_axis = 0
                
                # Normalize both segments to prevent clipping
                # Peak normalization to 0.95 to leave headroom
                max_val_1 = np.max(np.abs(y1_seg))
                if max_val_1 > 0:
                    y1_seg = y1_seg * (0.95 / max_val_1)
                    
                max_val_2 = np.max(np.abs(y2_seg))
                if max_val_2 > 0:
                    y2_seg = y2_seg * (0.95 / max_val_2)
                
                combined_audio = np.concatenate([y1_seg, y2_seg], axis=concat_axis)
                
                # Get Tempos
                try:
                    bpm1 = int(round(float(track1.tempo)))
                except (ValueError, TypeError):
                    bpm1 = "unk"
                    
                try:
                    bpm2 = int(round(float(track2.tempo)))
                except (ValueError, TypeError):
                    bpm2 = "unk"
                
                base_filename = f"{i}_pop{bpm1}bpm_{genre}{bpm2}bpm"
                
                # --- Annotations ---
                # Track 1
                t1_times = track1.beats.times
                t1_mask = (t1_times >= segment_start) & (t1_times <= segment_end)
                t1_times_filt = t1_times[t1_mask]
                
                t1_times_shifted = t1_times_filt - segment_start
                
                # Track 2
                t2_times = track2.beats.times
                t2_mask = (t2_times >= segment_start) & (t2_times <= segment_end)
                t2_times_filt = t2_times[t2_mask]

                if concat_axis == 1: 
                    dur1 = y1_seg.shape[1] / sr1
                else: 
                    dur1 = y1_seg.shape[0] / sr1
                
                t2_times_shifted = (t2_times_filt - segment_start) + dur1
                
                combined_times = np.concatenate([t1_times_shifted, t2_times_shifted])
                
                # --- Save Files (Atomic-ish: save only if all processing succeeded) ---
                
                # Validate audio data before writing
                if not np.isfinite(combined_audio).all():
                    raise ValueError(f"Audio contains NaN or inf values")
                if combined_audio.size == 0:
                    raise ValueError(f"Audio is empty")
                
                # Save audio
                out_wav_path = os.path.join(current_genre_audio_dir, f"{base_filename}.wav")
                try:
                    if combined_audio.ndim == 2 and combined_audio.shape[0] < combined_audio.shape[1]:
                        sf.write(out_wav_path, combined_audio.T, sr1)
                    else:
                        sf.write(out_wav_path, combined_audio, sr1)
                except Exception as write_err:
                    raise RuntimeError(f"Failed to write audio file (disk full or permissions issue?): {write_err}")
                
                # Save annotations
                np.save(os.path.join(current_genre_annot_dir, f"{base_filename}_times.npy"), combined_times)
                
            except Exception as e:
                print(f"Error processing pair {i} ({genre}): {e}")
                continue
            
    print("Cross-dataset generation complete.")
