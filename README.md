# F0Estimator


A neural network estimating the dominant melody, based on this 
[paper](https://www.researchgate.net/profile/Guillaume_Doras/publication/332434939_On_the_Use_of_U-Net_for_Dominant_Melody_Estimation_in_Polyphonic_Music/links/5cff5dbc4585157d15a20f9a/On-the-Use-of-U-Net-for-Dominant-Melody-Estimation-in-Polyphonic-Music.pdf).

For each audio file, it computes its HCQT and estimates its dominant melody. This representation 
is trimmed along the frequency axis to keep only 3 octaves around the mean pitch. Final resolution is one bin per semi-tone and each time frame is ~58 ms.

To use it:

    python -m f0estimator.factory --audio your_audio_directory_path --save where_to_save_directory_path

If you use it, please cite our work:

      
      @inproceedings{doras2019use,
        title={On the use of u-net for dominant melody estimation in polyphonic music},
        author={Doras, Guillaume and Esling, Philippe and Peeters, Geoffroy},
        booktitle={2019 International Workshop on Multilayer Music Representation and Processing (MMRP)},
        pages={66--70},
        year={2019},
        organization={IEEE}
      }
      