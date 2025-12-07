import librosa
import librosa.display
import matplotlib.pyplot as plt


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
    