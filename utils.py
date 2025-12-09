import numpy as np
import mir_eval
from IPython.display import Audio
import os
import librosa
import librosa.display
import ast
import matplotlib.pyplot as plt
import json
import pandas as pd


def plot_beats(audio_file, ground_truth, beats_bt, beats_bn, beats_mm, method, save=False):
    """Plot all three models"""
    
    output_dir = os.path.join(f"./results/beat_plots/" + method)
    os.makedirs(output_dir, exist_ok=True)
    
    y, sr = librosa.load(audio_file)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Model 1: beat_this
    ax = axes[0]
    librosa.display.waveshow(y=y, sr=sr, ax=ax, alpha=0.6, color='gray')
    ax.vlines(beats_bt, ymin=y.min(), ymax=y.max(), 
              color='red', linestyles='dashed', alpha=0.7, linewidth=1.5,
              label='beats')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Model 1: beat_this', fontweight='bold')
    ax.legend(loc='upper right')
    
    # Model 2: BeatNet
    ax = axes[1]
    librosa.display.waveshow(y=y, sr=sr, ax=ax, alpha=0.6, color='gray')
    ax.vlines(beats_bn, ymin=y.min(), ymax=y.max(), 
              color='blue', linestyles='dashed', alpha=0.7, linewidth=1.5,
              label='beats')
    ax.set_ylabel('Amplitude')
    ax.set_title('Model 2: BeatNet', fontweight='bold')
    ax.legend(loc='upper right')
    
    # Model 3: madmom
    ax = axes[2]
    librosa.display.waveshow(y=y, sr=sr, ax=ax, alpha=0.6, color='gray')
    ax.vlines(beats_mm, ymin=y.min(), ymax=y.max(), 
              color='green', linestyles='dashed', alpha=0.7, linewidth=1.5,
              label='beats')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Model 3: madmom', fontweight='bold')
    ax.legend(loc='upper right')
    
    # Model 4: ground truth
    ax = axes[3]
    librosa.display.waveshow(y=y, sr=sr, ax=ax, alpha=0.6, color='gray')
    ax.vlines(ground_truth, ymin=y.min(), ymax=y.max(), 
              color='black', linestyles='dashed', alpha=0.7, linewidth=1.5,
              label=f'beats')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Model 4: annotated beats', fontweight='bold')
    ax.legend(loc='upper right')
    
    file_name = os.path.basename(audio_file).replace('.wav', '')
    
    plt.tight_layout()
    if save == True:
        plt.savefig(os.path.join(output_dir, f'{file_name}_beat_comparison.png'), dpi=150)
        # print("\n✓ Saved: beat_comparison.png")
    plt.close(fig)
    
    
def mean_std(json_file):
    # Load JSON results file
    with open(json_file, 'r') as f:
        results = json.load(f)

    # Extract metrics for each model
    beat_this_scores = {'F-measure': [], 'Cemgil': [], 'CMLc': []}
    beat_net_scores = {'F-measure': [], 'Cemgil': [], 'CMLc': []}
    madmom_scores = {'F-measure': [], 'Cemgil': [], 'CMLc': []}

    for track_id, track_results in results.items():
        # Parse string-formatted dictionaries
        beat_this = eval(track_results['beat_this'])
        beat_net = eval(track_results['beat_net'])
        madmom = eval(track_results['madmom'])
        
        for metric in ['F-measure', 'Cemgil', 'CMLc']:
            beat_this_scores[metric].append(float(beat_this[metric]))
            beat_net_scores[metric].append(float(beat_net[metric]))
            madmom_scores[metric].append(float(madmom[metric]))

    # Calculate statistics
    stats = pd.DataFrame({
        'Model': ['beat_this', 'beat_this', 'beat_this', 'beat_net', 'beat_net', 'beat_net', 'madmom', 'madmom', 'madmom'],
        'Metric': ['F-measure', 'Cemgil', 'CMLc'] * 3,
        'Mean': [
            np.mean(beat_this_scores['F-measure']), np.mean(beat_this_scores['Cemgil']), np.mean(beat_this_scores['CMLc']),
            np.mean(beat_net_scores['F-measure']), np.mean(beat_net_scores['Cemgil']), np.mean(beat_net_scores['CMLc']),
            np.mean(madmom_scores['F-measure']), np.mean(madmom_scores['Cemgil']), np.mean(madmom_scores['CMLc'])
        ],
        'Std': [
            np.std(beat_this_scores['F-measure']), np.std(beat_this_scores['Cemgil']), np.std(beat_this_scores['CMLc']),
            np.std(beat_net_scores['F-measure']), np.std(beat_net_scores['Cemgil']), np.std(beat_net_scores['CMLc']),
            np.std(madmom_scores['F-measure']), np.std(madmom_scores['Cemgil']), np.std(madmom_scores['CMLc'])
        ]
    })

    print(stats)
    
    
METRIC_ORDER = ["F-measure", "Cemgil", "CMLc", "CMLt", "AMLc", "AMLt"]   
    
def parse_results(json_file):
    """Return {metric: {model: [values]}} from the stored evaluation JSON."""
    with open(json_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    metric_scores = {
        metric: {} for metric in METRIC_ORDER
    }

    for track_scores in raw_data.values():
        for model_name, metrics_str in track_scores.items():
            metrics = ast.literal_eval(metrics_str)
            for metric_name, value in metrics.items():
                metric_scores.setdefault(metric_name, {})
                metric_scores[metric_name].setdefault(model_name, [])
                metric_scores[metric_name][model_name].append(float(value))

    return metric_scores


def plot_results(json_file_name, save_path="./results/box_plots"):
    """Generate per-metric boxplots comparing models."""
    
    os.makedirs(save_path, exist_ok=True)
    results_base_dir = "./results/"
    
    json_file = os.path.join(results_base_dir + json_file_name)
    metric_scores = parse_results(json_file)

    n_metrics = len(METRIC_ORDER)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)

    for idx, metric in enumerate(METRIC_ORDER):
        ax = axes[idx] if n_metrics > 1 else axes
        model_names = sorted(metric_scores.get(metric, {}).keys())
        data = [metric_scores[metric][name] for name in model_names]
        ax.boxplot(data, labels=model_names, patch_artist=True)
        ax.set_title(metric)
        # ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # fig.suptitle(f"MIREX Scores • {json_file.stem}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    json_name = os.path.basename(json_file_name).replace(".json","")
    
    out_dir = os.path.join(save_path,f"{json_name}_boxplots.png")

    fig.savefig(out_dir, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # return save_path


def plot_combine_cross_genre_results(genres, save_path="./results/box_plots"):
    """Generate combined boxplots for all cross-genre experiments."""
    
    os.makedirs(save_path, exist_ok=True)
    results_base_dir = "./results/"
    
    n_genres = len(genres)
    n_metrics = len(METRIC_ORDER)
    
    # Create subplots: rows = genres, cols = metrics
    fig, axes = plt.subplots(n_genres, n_metrics, figsize=(5 * n_metrics, 4 * n_genres))
    
    for genre_idx, genre in enumerate(genres):
        json_file_name = f"cross_genre_{genre}_results.json"
        json_file = os.path.join(results_base_dir, json_file_name)
        
        try:
            metric_scores = parse_results(json_file)
            
            for metric_idx, metric in enumerate(METRIC_ORDER):
                ax = axes[genre_idx, metric_idx]
                model_names = sorted(metric_scores.get(metric, {}).keys())
                data = [metric_scores[metric][name] for name in model_names]
                ax.boxplot(data, labels=model_names, patch_artist=True)
                
                # Only show metric name on top row
                if genre_idx == 0:
                    ax.set_title(metric, fontweight='bold')
                
                # Only show genre name on first column
                if metric_idx == 0:
                    ax.set_ylabel(genre.capitalize(), fontweight='bold', fontsize=12)
                
                ax.set_ylim(0, 1)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                
        except FileNotFoundError:
            print(f"Warning: {json_file_name} not found")
            continue
    
    fig.suptitle("Cross-Genre Beat Tracking Results", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    out_dir = os.path.join(save_path, "cross_genre_combined_results.png")
    fig.savefig(out_dir, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to: {out_dir}")


def sonify_track(audio_file_path, beat_times, sample_rate=44100, hop_length=512):
    
    y, sr = librosa.load(audio_file_path, sr=sample_rate)
    
    if y is not None and len(beat_times) > 0:
        print("\n--- Sonifying Detected Pulse ---")
        try:
            # Generate click track from beat_times (in seconds)
            click_track = mir_eval.sonify.clicks(beat_times, 
                                                fs=sr, 
                                                length=len(y))
            
            # Mix original audio with click track
            mixed_audio = (y * 0.5) + (click_track * 0.5)
            
            # Normalize to prevent clipping
            mixed_audio_normalized = librosa.util.normalize(mixed_audio)
            
            # Display the audio player in your notebook
            print("Playing audio with detected beats (click track)...")
            return Audio(mixed_audio_normalized, rate=sr)
        
        except Exception as e:
            print(f"Error during sonification: {e}")
            pass