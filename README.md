# F0Estimator


A neural network estimating the dominant melody, based on this 
[paper](https://www.researchgate.net/profile/Guillaume_Doras/publication/332434939_On_the_Use_of_U-Net_for_Dominant_Melody_Estimation_in_Polyphonic_Music/links/5cff5dbc4585157d15a20f9a/On-the-Use-of-U-Net-for-Dominant-Melody-Estimation-in-Polyphonic-Music.pdf).

For each audio file, tt computes its HCQT, estimates its dominant melody and finally frequency span and resolution
to 3 octaves around the mean pitch and one bin per semi-tone. 

To use it:

    python factory.py --audio your_audio_directory_path --save where_to_save_directory_path

If you use it, please cite our work:

      
      @inproceedings{doras2019use,
        title={On the use of u-net for dominant melody estimation in polyphonic music},
        author={Doras, Guillaume and Esling, Philippe and Peeters, Geoffroy},
        booktitle={2019 International Workshop on Multilayer Music Representation and Processing (MMRP)},
        pages={66--70},
        year={2019},
        organization={IEEE}
      }
