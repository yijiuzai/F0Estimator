import tensorflow as tf
import numpy as np
import os
import glob
import tempfile

from f0estimator.model.f0_unet_runner import F0ModelRunner

import f0estimator.helpers.utils as utils


class F0UnetRunnerTest(tf.test.TestCase):



	def test_f0_runner(self):

		model_directory_path = './tests_checkpoints/f0' # this is a fake model (config has been modified)


		runner = F0ModelRunner(model_directory_path=model_directory_path)
		runner.build_model()

		hparams = runner.model.hparams
		input_chunk_shape = hparams.database_src_data_shape # [t, f, h] # small t to speed up test on cpu, f and h is imposed by this model

		with tempfile.TemporaryDirectory() as save_directory_path:

			# create dummy data that will fit this model
			source_durations = [9, 11]

			src_filepaths = []

			for i, source_duration in enumerate(source_durations):

				input_data = np.random.random([input_chunk_shape[2], input_chunk_shape[1], source_duration]) # [h, f, t]
				input_data = input_data.astype(np.float32)
				src_data_filepath = os.path.join(save_directory_path, '%d.hcqt.npy.gz' % i)
				utils.save_data(src_data_filepath, input_data)

				src_filepaths.append(src_data_filepath)

			# process
			runner.apply_model(src_filepaths, save_directory_path=save_directory_path)

			# check files
			tgt_data_files_extension = 'f0.npy'
			tgt_filepaths = glob.glob(os.path.join(save_directory_path, '*.%s' % tgt_data_files_extension))

			self.assertEqual(len(tgt_filepaths), len(src_filepaths))

			for src_filepath, tgt_filepath in zip(src_filepaths, tgt_filepaths):
				src_data = utils.load_data(src_filepath) # [h, f, t]
				tgt_data = utils.load_data(tgt_filepath) # [t, f]
				self.assertEqual(tgt_data.ndim, 2)
				self.assertEqual((src_data.shape[2], src_data.shape[1]), tgt_data.shape) # target is [t, f], src is [h, f, t]
				self.assertEqual(src_data.dtype, np.float32)
				self.assertEqual(tgt_data.dtype, np.float32)