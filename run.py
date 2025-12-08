import mir_eval
import os
import madmom
import numpy as np
import json
from beat_this.inference import File2Beats
from BeatNet.BeatNet import BeatNet

from utils import plot_beats

def evaluate_beats(ref_beats, est_beats):
    fscore = mir_eval.beat.f_measure(ref_beats, est_beats)
    cemgil_score = mir_eval.beat.cemgil(ref_beats, est_beats)
    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(ref_beats, est_beats)
    results = {
        "F-measure": f"{fscore:.2f}",
        "Cemgil": f"{float(cemgil_score[0]):.2f}",
        "CMLc": f"{float(cmlc):.2f}",
        "CMLt": f"{float(cmlt):.2f}",
        "AMLc": f"{float(amlc):.2f}",
        "AMLt": f"{float(amlt):.2f}"
    }
    return results

def run_model(audio_file, plot=False, plot_save=False):

    # Model 1: beat_this
    beat_this = File2Beats(checkpoint_path="final0", device="cpu", dbn=True)
    beats_bt, downbeats_bt = beat_this(audio_file)
    # print(f"{len(beats_bt)} beats from beat_this")  

    # Model 2: BeatNet
    beatnet = BeatNet(model=1, mode='offline', inference_model='DBN', plot=[], thread=False)
    output_bn = beatnet.process(audio_file)
    beats_bn = output_bn[:, 0] if output_bn is not None else np.array([])
    # print(f"{len(beats_bn)} beats from BeatNet")

    # Model 3: madmom
    rnn_proc = madmom.features.beats.RNNBeatProcessor()
    dbn_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = rnn_proc(audio_file)
    beats_mm = dbn_proc(act)
    # print(f"{len(beats_mm)} beats from Madmom")

    # Plot
    if plot == True:
        plot_beats(audio_file, beats_bt, beats_bn, beats_mm, save=plot_save)
    
    return beats_bt, beats_bn, beats_mm

def evaluation(file_name, ref_beats, est_bt, est_bn, est_mm, save=True, json_file='results/results.json'):
    
    beat_this_results = evaluate_beats(ref_beats, est_bt)
    beat_net_results = evaluate_beats(ref_beats, est_bn)
    madmom_results = evaluate_beats(ref_beats, est_mm)
    
    combined_results = {
        "beat_this": f"{beat_this_results}",
        "beat_net": f"{beat_net_results}",
        "madmom": f"{madmom_results}"
    }

    info = {f"{file_name}": combined_results}
    
    os.makedirs("results", exist_ok=True)
    base_dir = "./results/"
    
    if save==True:
        try:
            with open(os.path.join(base_dir +json_file), 'r') as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = {} 
            
        all_results.update(info)
        
        with open(os.path.join(base_dir + json_file), 'w') as f:
            json.dump(all_results, f, indent=2)
    
    return combined_results, info
