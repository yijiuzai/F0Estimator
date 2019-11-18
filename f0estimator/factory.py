import os
import glob

import f0estimator.helpers.utils as utils


F0_LOGGER_NAME = 'f0s'
F0_FILES_EXTENSION = 'f0.npy'



def compute_and_save_f0s_for_audio_directory_paths(audio_directory_paths,
                                                   save_directory_path,
                                                   audio_files_extensions='mp3',
                                                   audio_sampling_rate_in_hz=22050,
                                                   hcqt_hop_length_in_bins=256,
                                                   hcqt_frequency_min_in_hz=32.7,
                                                   hcqt_num_octaves=6,
                                                   hcqt_num_bins_per_octave=60,
                                                   hcqt_harmonics=(0.5, 1, 2, 3, 4, 5),
                                                   n_hcqt_jobs=1,
                                                   n_jobs=16):
	"""
	Computes HCQTs for all audio files in this directory.

	Args:
		audio_directory_paths: list or str, the directory(ies) of the audio files.
		save_directory_path: str, where to save the target files.
		audio_files_extensions: list or str, the extension(s) of the audio files to consider in this directory.
		audio_sampling_rate_in_hz:
		hcqt_hop_length_in_bins:
		hcqt_frequency_min_in_hz:
		hcqt_num_octaves:
		hcqt_num_bins_per_octave:
		hcqt_harmonics:
		n_hcqt_jobs:
		n_jobs:

	Returns:

	"""
	if type(audio_directory_paths) is str:
		audio_directory_paths = [audio_directory_paths]

	for audio_directory_path in audio_directory_paths:
		if not os.path.exists(audio_directory_path):
			raise ValueError('There is no directory at %s.' % audio_directory_path)

	if type(audio_files_extensions) is str:
		audio_files_extensions = [audio_files_extensions]

	# get all audio files in this directory
	audio_filepaths = []
	for audio_directory_path in audio_directory_paths:
		for audio_files_extension in audio_files_extensions:
			audio_filepaths += glob.glob(os.path.join(audio_directory_path, '*.%s' % audio_files_extension))

	if not os.path.exists(save_directory_path):
		os.makedirs(save_directory_path)

	utils.get_logger(F0_LOGGER_NAME, logs_directory_path=save_directory_path)

	audio_and_save_filepaths = [(audio_filepath,
	                             generate_f0_filepath_for_audio_filepath(audio_filepath, save_directory_path))
	                            for audio_filepath in audio_filepaths]

	compute_and_save_f0s_for_audio_and_save_filepaths(audio_and_save_filepaths,
	                                                    audio_sampling_rate_in_hz=audio_sampling_rate_in_hz,
	                                                    hcqt_hop_length_in_bins=hcqt_hop_length_in_bins,
	                                                    hcqt_frequency_min_in_hz=hcqt_frequency_min_in_hz,
	                                                    hcqt_num_octaves=hcqt_num_octaves,
	                                                    hcqt_num_bins_per_octave=hcqt_num_bins_per_octave,
	                                                    hcqt_harmonics=hcqt_harmonics,
	                                                    n_hcqt_jobs=n_hcqt_jobs,
	                                                    n_jobs=n_jobs)



def generate_f0_filepath_for_audio_filepath(audio_filepath, save_directory_path):

	return os.path.join(save_directory_path, '%s.%s' % (os.path.splitext(os.path.basename(audio_filepath))[0], F0_FILES_EXTENSION))



def compute_and_save_f0s_for_audio_and_save_filepaths(audio_and_save_filepaths,
                                                      audio_sampling_rate_in_hz=22050,
                                                      hcqt_hop_length_in_bins=256,
                                                      hcqt_frequency_min_in_hz=32.7,
                                                      hcqt_num_octaves=6,
                                                      hcqt_num_bins_per_octave=60,
                                                      hcqt_harmonics=(0.5, 1, 2, 3, 4, 5),
                                                      n_hcqt_jobs=1,
                                                      n_jobs=16):
	"""
	Computes and save representation for audio/save filepaths pairs.

	Args:
		audio_and_save_filepaths: list, pairs of audio_filepath and save_filepath
		audio_sampling_rate_in_hz:
		hcqt_hop_length_in_bins:
		hcqt_frequency_min_in_hz:
		hcqt_num_octaves:
		hcqt_num_bins_per_octave:
		hcqt_harmonics:
		n_hcqt_jobs:
		n_jobs:

	Returns:

	"""

	logger = utils.get_logger(F0_LOGGER_NAME)

	logger.info('Computing files for %d audio files... ' % len(audio_and_save_filepaths))

	# keep only audio for which the hcqt has not been computed already
	logger.info('Checking already computed files... ')
	remaining_audio_and_save_filepaths = [(audio_filepath, save_filepath)
	                                      for (audio_filepath, save_filepath) in audio_and_save_filepaths
	                                      if not os.path.exists(save_filepath)]

	logger.info('Computing files for %d remaining audio files... ' % len(remaining_audio_and_save_filepaths))

	if len(remaining_audio_and_save_filepaths) > 0:
		if n_jobs > 1:

			Parallel(n_jobs=n_jobs, verbose=5)(
				delayed(compute_and_save_log_hcqt_for_audio_filepath)(
					audio_and_save_filepaths[0],
					audio_and_save_filepaths[1],
					audio_sampling_rate_in_hz=audio_sampling_rate_in_hz,
					hcqt_hop_length_in_bins=hcqt_hop_length_in_bins,
					hcqt_frequency_min_in_hz=hcqt_frequency_min_in_hz,
					hcqt_num_octaves=hcqt_num_octaves,
					hcqt_num_bins_per_octave=hcqt_num_bins_per_octave,
					hcqt_harmonics=hcqt_harmonics,
					n_hcqt_jobs=n_hcqt_jobs,
				) for audio_and_save_filepaths in remaining_audio_and_save_filepaths)

		else:

			for audio_and_save_filepaths in remaining_audio_and_save_filepaths:
				compute_and_save_log_hcqt_for_audio_filepath(
					audio_and_save_filepaths[0],
					audio_and_save_filepaths[1],
					audio_sampling_rate_in_hz=audio_sampling_rate_in_hz,
					hcqt_hop_length_in_bins=hcqt_hop_length_in_bins,
					hcqt_frequency_min_in_hz=hcqt_frequency_min_in_hz,
					hcqt_num_octaves=hcqt_num_octaves,
					hcqt_num_bins_per_octave=hcqt_num_bins_per_octave,
					hcqt_harmonics=hcqt_harmonics,
					n_hcqt_jobs=n_hcqt_jobs)

		logger.info('Computed and saved %d hcqt files.' % (len(remaining_audio_and_save_filepaths)))

	logger.info('Done. That\'s all folks!')