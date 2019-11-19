import tensorflow as tf
import tempfile
import glob
import os
import f0estimator.factory as f0_factory



class F0RunnerTest(tf.test.TestCase):


	def test_compute_and_save_f0s_for_audio_directory_paths(self):

		model_directory_path = './model/tests_checkpoints/f0' # this is a fake model (config has been modified)
		audio_directory_path = '.'

		with tempfile.TemporaryDirectory() as save_directory_path:

			f0_factory.compute_and_save_f0s_for_audio_directory_paths(audio_directory_paths=[audio_directory_path],
			                                                          save_directory_path=save_directory_path,
			                                                          model_directory_path=model_directory_path,
			                                                          audio_files_extensions='mp3',
			                                                          audio_sampling_rate_in_hz=22050,
			                                                          hcqt_hop_length_in_bins=8192,
			                                                          hcqt_frequency_min_in_hz=32.7,
			                                                          hcqt_num_octaves=6,
			                                                          hcqt_num_bins_per_octave=12,
			                                                          hcqt_harmonics=(0.5, 1, 2, 3, 4, 5),
			                                                          n_hcqt_jobs=1,
			                                                          n_devices=1)

			mp3_filepaths = glob.glob(os.path.join(audio_directory_path, '*.mp3'))
			f0_filepaths = glob.glob(os.path.join(save_directory_path, '*.%s' % f0_factory.F0_FILES_EXTENSION))

			self.assertEqual(len(f0_filepaths), len(mp3_filepaths))

