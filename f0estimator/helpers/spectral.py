import argparse
import os
from joblib import Parallel, delayed
import glob
import numpy as np
import librosa

CORPUS_HCQT_FILES_EXTENSION = 'hcqt.npy.gz'





#-----------------------------------------------------------------------------------------------------------------------
# Hcqt
#-----------------------------------------------------------------------------------------------------------------------

def compute_and_save_log_hcqt_for_audio_filepath(audio_filepath,
                                                 hcqt_filepath=None,
                                                 audio_sampling_rate_in_hz=22050,
                                                 hcqt_hop_length_in_bins=256,
                                                 hcqt_frequency_min_in_hz=32.7,
                                                 hcqt_num_octaves=6,
                                                 hcqt_num_bins_per_octave=60,
                                                 hcqt_harmonics=(0.5, 1, 2, 3, 4, 5),
                                                 n_hcqt_jobs=1,
                                                 ):

	if type(hcqt_harmonics) is tuple:
		hcqt_harmonics = list(hcqt_harmonics)


	if hcqt_filepath is None: # by default, we save the hcqt beside the audio
		hcqt_filepath ='%s.%s' % (audio_filepath[:-4], CORPUS_HCQT_FILES_EXTENSION)

	if not os.path.exists(hcqt_filepath):

		hcqt = compute_log_hcqt(audio_filepath,
		                        audio_sampling_rate_in_hz=audio_sampling_rate_in_hz,
		                        hcqt_hop_length_in_bins=hcqt_hop_length_in_bins,
		                        hcqt_frequency_min_in_hz=hcqt_frequency_min_in_hz,
		                        hcqt_num_octaves=hcqt_num_octaves,
		                        hcqt_num_bins_per_octave=hcqt_num_bins_per_octave,
		                        hcqt_harmonics=hcqt_harmonics,
		                        n_hcqt_jobs=n_hcqt_jobs)


		save_data(hcqt_filepath, hcqt) # [h, f, t]

		del hcqt

	return hcqt_filepath




def compute_log_hcqt(audio_filepath,
                     audio_sampling_rate_in_hz=22050,
                     hcqt_hop_length_in_bins=256,
                     hcqt_frequency_min_in_hz=32.7,
                     hcqt_num_octaves=6,
                     hcqt_num_bins_per_octave=60,
                     hcqt_harmonics=(0.5, 1, 2, 3, 4, 5),
                     n_hcqt_jobs=1
                     ):
	"""
	Compute the harmonic CQT from a given audio file.

	Args:
		audio_filepath: str, the path to the audio file
		audio_sampling_rate_in_hz: int, the rate to sample the audio data
		hcqt_hop_length_in_bins: int, hcqt hop size
		hcqt_frequency_min_in_hz: float, minimal HCQT frequency
		hcqt_num_octaves: int, the number of HCQT octaves
		hcqt_num_bins_per_octave: int, the number of bins per octaves.
		hcqt_harmonics: int, the harmonics factor to the f_min
		n_hcqt_jobs: int, the number of jobs to use to compute the HCQT.

	Returns:
		log_hcqt: ndarray, the HCQT of shape shape [h, f, t] and dtype np.float32

	"""

	audio_data, sampling_rate_in_hz = librosa.core.load(audio_filepath,
	                                                    sr=audio_sampling_rate_in_hz,
	                                                    mono=True)

	harmonics = hcqt_harmonics if hcqt_harmonics else [0.5,1,2,3,4,5]

	cqt_list = []
	shapes = []

	if n_hcqt_jobs > 1:

		cqt_list = Parallel(n_jobs=n_hcqt_jobs, verbose=5)(
			delayed(librosa.cqt)(
				audio_data,
				sr=sampling_rate_in_hz,
				hop_length=hcqt_hop_length_in_bins,
				fmin=hcqt_frequency_min_in_hz*float(h),
				n_bins=hcqt_num_bins_per_octave*hcqt_num_octaves,
				bins_per_octave=hcqt_num_bins_per_octave
			) for h in harmonics)

		cqt_list = [np.abs(cqt) for cqt in cqt_list]
		shapes = [cqt.shape for cqt in cqt_list]

	else:
		for h in harmonics:

			cqt = librosa.cqt(
				audio_data,
				sr=sampling_rate_in_hz,
				hop_length=hcqt_hop_length_in_bins,
				fmin=hcqt_frequency_min_in_hz*float(h),
				n_bins=hcqt_num_bins_per_octave*hcqt_num_octaves,
				bins_per_octave=hcqt_num_bins_per_octave
			)

			cqt = np.abs(cqt) # in order to gain memory

			cqt_list.append(cqt)
			shapes.append(cqt.shape)

	del audio_data

	shapes_equal = [s == shapes[0] for s in shapes]
	if not all(shapes_equal):
		min_time = np.min([s[1] for s in shapes])
		new_cqt_list = []
		for i in range(len(cqt_list)):
			cqt = cqt_list.pop(0)
			new_cqt_list.append(cqt[:, :min_time]) # originally new_cqt_list.append(cqt_list[i][:, :min_time])
			del cqt

		cqt_list = new_cqt_list

	log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
		np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

	log_hcqt = log_hcqt.astype(np.float32)

	return log_hcqt



