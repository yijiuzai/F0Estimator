import os
import glob
import argparse
import tempfile
import time

from f0estimator.model.f0_unet_runner import F0ModelRunner
import f0estimator.helpers.spectral as spectral
import f0estimator.helpers.utils as utils


F0_LOGGER_NAME = 'f0s'
HCQT_FILES_EXTENSION = 'hcqt.npy.gz'
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

	compute_and_save_f0s_for_audio_filepaths(audio_filepaths,
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

def generate_hcqt_filepath_for_audio_filepath(audio_filepath, save_directory_path):

	return os.path.join(save_directory_path, '%s.%s' % (os.path.splitext(os.path.basename(audio_filepath))[0], HCQT_FILES_EXTENSION))


def compute_and_save_f0s_for_audio_filepaths(audio_filepaths,
                                                      save_directory_path,
                                                      audio_sampling_rate_in_hz=22050,
                                                      hcqt_hop_length_in_bins=256,
                                                      hcqt_frequency_min_in_hz=32.7,
                                                      hcqt_num_octaves=6,
                                                      hcqt_num_bins_per_octave=60,
                                                      hcqt_harmonics=(0.5, 1, 2, 3, 4, 5),
                                                      n_hcqt_jobs=1,
                                                      n_devices=1):
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



	logger.info('Computing files for %d audio files... ' % len(audio_filepaths))

	# keep only audio for which the hcqt has not been computed already
	logger.info('Checking already computed files... ')
	audio_and_save_filepaths = [(audio_filepath,
	                             generate_f0_filepath_for_audio_filepath(audio_filepath, save_directory_path))
	                            for audio_filepath in audio_filepaths]

	remaining_audio_filepaths = [audio_filepath
	                             for (audio_filepath, save_filepath) in audio_and_save_filepaths
	                             if not os.path.exists(save_filepath)]

	logger.info('Computing files for %d remaining audio files... ' % len(remaining_audio_filepaths))

	num_remaining_audio_filepaths = len(remaining_audio_filepaths)

	if num_remaining_audio_filepaths > 0:

		start_time = time.time()

		logger.info('Loading model... ')
		devices = utils.acquire_devices()

		runner = F0ModelRunner(devices=devices)
		runner.build_model()

		with tempfile.TemporaryDirectory() as tmp_directory_path:

			for i, audio_filepath in enumerate(remaining_audio_filepaths):

				# first, compute the HCQT
				hcqt_filepath = generate_hcqt_filepath_for_audio_filepath(audio_filepath,
				                                                          save_directory_path=tmp_directory_path)

				spectral.compute_and_save_log_hcqt_for_audio_filepath(audio_filepath,
				                                                      hcqt_filepath=hcqt_filepath,
				                                                      audio_sampling_rate_in_hz=audio_sampling_rate_in_hz,
				                                                      hcqt_hop_length_in_bins=hcqt_hop_length_in_bins,
				                                                      hcqt_frequency_min_in_hz=hcqt_frequency_min_in_hz,
				                                                      hcqt_num_octaves=hcqt_num_octaves,
				                                                      hcqt_num_bins_per_octave=hcqt_num_bins_per_octave,
				                                                      hcqt_harmonics=hcqt_harmonics,
				                                                      n_hcqt_jobs=n_hcqt_jobs)

				# run the model
				runner.apply_model([hcqt_filepath],
				                   save_directory_path=save_directory_path)


				os.remove(hcqt_filepath)

				hop = max(10, num_remaining_audio_filepaths // 10)
				if i > 0 and i % hop == 0:
					logger.info("  Applied model to %d (out of %d) examples (eta: %s)..." %
					            (i,
					             num_remaining_audio_filepaths,
					             utils.eta_based_on_elapsed_time(i, num_remaining_audio_filepaths, start_time)))


		logger.info('Computed and saved %d files.' % num_remaining_audio_filepaths)

	logger.info('Done. That\'s all folks!')










def main():

	# get the directory path where to save the corpus out of the arguments
	arguments_parser = argparse.ArgumentParser()

	arguments_parser.add_argument("--save", type=str, default=None, required=True, help="A directory where to save the produced files.")
	arguments_parser.add_argument("--audio", type=str, nargs='+',default=None, help="A directory from where to produce files.")

	arguments_parser.add_argument("--sr", type=int, default=22050, help="Audio sampling rate in Hz..")
	arguments_parser.add_argument("--hop", type=int, default=256, help="Hop size in bins.")
	arguments_parser.add_argument("--no", type=int, default=6, help="Num octaves.")
	arguments_parser.add_argument("--nbpo", type=int, default=60, help="Num bins per octave.")
	arguments_parser.add_argument("--hs", type=int, nargs='+', default=(0.5, 1, 2, 3, 4, 5), help="Hcqt harmonics.")

	arguments_parser.add_argument("--n_jobs",  type=int, default=1, help="Num processes.")



	flags, _ = arguments_parser.parse_known_args()

	compute_and_save_hcqts_for_audio_directory_paths(audio_directory_paths=flags.audio,
	                                                 save_directory_path=flags.save,
	                                                 audio_sampling_rate_in_hz=flags.sr,
	                                                 hcqt_hop_length_in_bins=flags.hop,
	                                                 hcqt_num_octaves=flags.no,
	                                                 hcqt_num_bins_per_octave=flags.nbpo,
	                                                 hcqt_harmonics=flags.hs,
	                                                 n_jobs=flags.n_jobs)





if __name__ == '__main__':
	main()