import mirdata
import shutil
import os
import numpy as np
import soundfile as sf

from data_preprocessing.time_stretch import load_dataset


def concatenate_tracks_single_genre(
    dataset_name, data_home, output_dir="./dataset", segment_start=0, segment_end=15
):
    """
    Creates a synthetic dataset by combining segments from pairs of tracks.

    Args:
        dataset: Initialized mirdata dataset (e.g. rwc_popular).
        output_dir: Directory to save the generated files.
        n_files: Number of synthetic files to generate (default 100).
        segment_start: Start time of segment in seconds (default 15).
        segment_end: End time of segment in seconds (default 30).
    """

    # Create output directories
    audio_dir = os.path.join(output_dir, f"single_genre/{dataset_name}/audio")
    annot_dir = os.path.join(output_dir, f"single_genre/{dataset_name}/annotations")

    # Clean up existing directories to ensure overwrite
    if os.path.exists(audio_dir):
        shutil.rmtree(audio_dir)
    if os.path.exists(annot_dir):
        shutil.rmtree(annot_dir)

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)

    dataset, track_ids = load_dataset(dataset_name, data_home)

    # Calculate number of pairs
    n_files = len(track_ids) // 2
    print(
        f"Generating {n_files} synthetic files in {output_dir}/single_genre/{dataset_name}..."
    )

    for i in range(n_files):
        # Select 2 unique tracks sequentially
        track1_id = track_ids[2 * i]
        track2_id = track_ids[2 * i + 1]

        track1 = dataset.track(track1_id)
        track2 = dataset.track(track2_id)

        # --- Audio Processing ---
        y1, sr1 = track1.audio
        y2, sr2 = track2.audio

        start_sample = int(segment_start * sr1)
        end_sample = int(segment_end * sr1)

        # Slice audio (handling bounds safely)
        if len(y1) < end_sample:
            y1_seg = y1[start_sample:]
        else:
            y1_seg = y1[start_sample:end_sample]

        # For track 2, assume similar SR or just slice by calculated indices
        # Ideally would check SR match, but keeping it simple as per original design
        if len(y2) < end_sample:
            if sr1 != sr2:
                # simple fallback if SR differs
                s2_start = int(segment_start * sr2)
                y2_seg = y2[s2_start:]
            else:
                y2_seg = y2[start_sample:]
        else:
            if sr1 != sr2:
                s2_start = int(segment_start * sr2)
                s2_end = int(segment_end * sr2)
                y2_seg = y2[s2_start:s2_end]
            else:
                y2_seg = y2[start_sample:end_sample]

        # Normalize both segments to prevent clipping
        # Peak normalization to 0.95 to leave headroom
        max_val_1 = np.max(np.abs(y1_seg))
        if max_val_1 > 0:
            y1_seg = y1_seg * (0.95 / max_val_1)

        max_val_2 = np.max(np.abs(y2_seg))
        if max_val_2 > 0:
            y2_seg = y2_seg * (0.95 / max_val_2)

        # Concatenate audio
        combined_audio = np.concatenate([y1_seg, y2_seg])

        bpm1 = track1.tempo
        bpm2 = track2.tempo

        base_filename = f"{i}_pop_bpm{bpm1}_pop_bpm{bpm2}"

        # Save audio
        out_wav_path = os.path.join(audio_dir, f"{base_filename}.wav")
        sf.write(out_wav_path, combined_audio, sr1)

        # --- Annotation Processing (Beats) ---
        # Track 1
        t1_times = track1.beats.times
        t1_mask = (t1_times >= segment_start) & (t1_times <= segment_end)

        t1_times_filt = t1_times[t1_mask]

        # Shift Track 1 timestamps to start at 0
        t1_times_shifted = t1_times_filt - segment_start

        # Track 2
        t2_times = track2.beats.times
        t2_mask = (t2_times >= segment_start) & (t2_times <= segment_end)

        t2_times_filt = t2_times[t2_mask]

        # Shift Track 2 timestamps to follow Track 1
        # The T2 segment starts after T1 segment (duration = segment_end - segment_start)
        # T2 beat time relative to start of T2 segment is (t - segment_start)
        # So in combined file: (t - segment_start) + (segment_end - segment_start)
        segment_duration = segment_end - segment_start
        t2_times_shifted = (t2_times_filt - segment_start) + segment_duration

        # combine
        combined_times = np.concatenate([t1_times_shifted, t2_times_shifted])

        # Save annotations
        np.save(os.path.join(annot_dir, f"{base_filename}_times.npy"), combined_times)

    print("Generation complete.")
