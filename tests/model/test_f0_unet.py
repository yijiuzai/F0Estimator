import tensorflow as tf
import numpy as np

from f0estimator.model.f0_unet import F0Unet


class Test(tf.test.TestCase):


	def test_f0_unet(self):

		model_directory_path = './tests_checkpoints/f0'

		# build model
		model = F0Unet(model_directory_path=model_directory_path)
		model.build()

		# create dummy data that will fit this model
		input_chunk_shape = model.hparams.database_src_data_shape # [t, f, h] # small t ti speed up test on cpu, f and h is imposed by this model
		input_chunk = np.random.random(input_chunk_shape)

		batch_size = 2 # whatever, but small t and h to speed up test on cpu,
		input_chunks = [input_chunk] * batch_size # [num_batch, t, ...]
		input_chunks = np.stack(input_chunks)  #[num_batch, t, ...]

		batched_output = model.apply(dataset_placeholders=[input_chunks]) # [num_batch, t, ...]

		self.assertIsNotNone(batched_output)
		self.assertEqual(batched_output.dtype, np.float32)
		self.assertEqual(batched_output.shape, tuple([batch_size] + input_chunk_shape[:-1]))


