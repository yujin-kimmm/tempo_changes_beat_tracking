# Evaluating Beat Detection Model Performance on Tempo Changes — NYU MIR Final Project Fall 2025

**Group 1:** Atilay Kucukoglu, Mercy Fang, Morris Chang, Yujin Kim  

---

This project investigates the research question:

> **How do state-of-the-art beat detection models perform on data with tempo changes?**

We address this by testing three models (madmom, Beat This!, BeatNet) on synthesized datasets consisting of music with tempo changes.

---

## Overview

*The project consists of the following steps:*

- Synthesizing datasets using three different methods to obtain audio files with tempo changes.
- Running the models on these curated datasets.
- Evaluating and discussing each model’s performance on each dataset.

---
## Dataset

Dataset should be downloaded from this [link](https://drive.google.com/drive/folders/1clNjw56TeBiIlDmy2Y4oC4cHq6GZPGca?usp=drive_link) and saved as "./dataset/" in this repository.
---

## Components

- **`notebook.ipynb`**  
  Main notebook that:
  - creates the datasets  
  - runs the models on these datasets  
  - saves the results  
  - generates the plots

- **`results.ipynb`**  
  Notebook that presents our results and discussion using the plots and sonifications.

- **`run.py`**  
  Contains functions for:
  - running the models  
  - evaluating the results  
  - saving them as JSON files

- **`utils.py`**  
  Contains helper functions for plotting and sonifying the results.

- **`results/`**  
  Contains:
  - precomputed results as JSON files  
  - box plots  
  - beat plots for each method

- **`data_preprocessing/`**  
  Collects the Python scripts used for synthesizing the datasets:
  - `baseline.py`: Creates the baseline dataset, which consists of 30 seconds of RWC_Popular samples without any processing.
  - `single_concat.py`: Concatenates samples from the RWC_Popular dataset. It concatenates 15 seconds of track 1 with 15 seconds of track 2, and continues in this manner until the end.
  - `cross_concat.py`: Concatenates samples from the RWC_Popular dataset with the GTZAN_genre dataset. It concatenates 15 seconds of track 1 of RWC_Popular with 15 seconds of track 1 from a given GTZAN genre, and continues similarly.
  - `time_stretch.py`: Creates samples that gradually increase or decrease in tempo using a time-stretching algorithm.

---

## Models Used

| Model      | Activation                                | Post-Processing                    |
|------------|-------------------------------------------|------------------------------------|
| madmom     | Recurrent Neural Networks (RNN)           | Dynamic Bayesian Networks (DBN)    |
| BeatNet    | Convolutional Recurrent Neural Network    | Dynamic Bayesian Networks (DBN)    |
| Beat This! | Convolution + Transformer                 | Dynamic Bayesian Networks (DBN)    |

---

## Dataset Creation

We created three datasets by synthesizing RWC_Popular and GTZAN_genre to test these models on tempo changes:

1. **Same-genre concatenation (RWC_Popular only)**  
   - We used the RWC_Popular dataset and concatenated the first 15 seconds of each sample with the next sample.  
   - This is referred to as the *same-genre concatenation* method.  
   - Since the RWC_Popular dataset consists of 100 total samples, concatenating track 1 with track 2, track 3 with track 4, and so on yields 50 samples for this dataset.

2. **Different-genre concatenation (RWC_Popular + GTZAN_genre)**  
   - We concatenated RWC_Popular with each genre in the GTZAN_genre dataset.  
   - This is referred to as the *different-genre concatenation* method.  
   - GTZAN_genre consists of 100 samples per 10 genres (1000 total). Each of these samples is concatenated with a sample from the RWC_Popular dataset to form a dataset with tempo and genre changes.

3. **Time-stretching (gradual tempo change)**  
   - We implemented a time-stretch algorithm and used it to gradually make each sample from the RWC_Popular dataset faster or slower.
   - This dataset consists of 100 samples.

All these datasets can be recreated either by running `notebook.ipynb` or by using the scripts inside the `data_preprocessing/` folder.

---

## Overall Results

- Looking at the F-measures and Cemgil scores, **madmom** consistently performs better than the other two models, except for the linear stretch method, where **Beat This!** performs best and madmom performs worst.
- For the linear-stretch method, examining **CMLc** and **AMLc** shows that madmom has the worst CMLc but the best AMLc. This large gap suggests that madmom often predicts beats at double-time or half-time relative to the ground truth.
- Overall, the models do **not** exhibit severe performance degradation under tempo changes.  
  - madmom has slightly higher accuracy compared to Beat This!  
  - Beat This! is more stable than madmom in terms of continuity.
  - BeatNet’s performance lies between madmom and Beat This!.

---

## Roles

- **Data Preprocessing:** Atilay, Yujin  
- **Setting up Models:** Mercy, Morris  
- **Results and Discussion:** Atilay, Yujin, Mercy, Morris  

---

We gratefully acknowledge the publicly available datasets, models, and libraries that made this project possible.
