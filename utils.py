import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import pandas as pd


def plot_beats(audio_file, beats_bt, beats_bn, beats_mm, save=False):
    """Plot all three models"""
    y, sr = librosa.load(audio_file)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Model 1: beat_this
    ax = axes[0]
    librosa.display.waveshow(y=y, sr=sr, ax=ax, alpha=0.6, color='gray')
    ax.vlines(beats_bt, ymin=y.min(), ymax=y.max(), 
              color='red', linestyles='dashed', alpha=0.7, linewidth=1.5,
              label='beats')
    
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
    
    plt.tight_layout()
    if save == True:
        plt.savefig('beat_comparison.png', dpi=150)
        print("\n✓ Saved: beat_comparison.png")
    plt.show()
    
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