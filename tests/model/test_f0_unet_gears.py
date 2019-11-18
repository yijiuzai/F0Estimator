import tensorflow as tf
import numpy as np

from f0estimator.model.f0_unet_gears import F0UnetGears
from f0estimator.helpers.utils import SimpleHParams


class Test(tf.test.TestCase):


	@classmethod
	def setUpClass(cls):
		pass

	def setUp(self):
		pass

	def tearDown(self):
		pass

	@classmethod
	def tearDownClass(cls):
		pass



	def test_f0_unet_gears(self):

		hparams = SimpleHParams(
			dict(

				# unet
				unet_floors_num_layers = [2, 2, 2, 2],
				unet_first_floor_layers_num_kernels = 2,
				unet_layers_kernel_shape = [3, 3],
				unet_layers_pool_shape = [2, 2],
				unet_layers_pool_stride = [2, 2],

			)
		)

		input_duration = 148 # whatever
		input_shape = [3, input_duration, 72, 2] # [b, t, f, h]

		# now we will create a dedicated graph for each configuration
		current_graph = tf.Graph()

		with current_graph.as_default():

			inputs = tf.random_uniform(input_shape)

			unet = F0UnetGears(hparams)
			outputs = unet.apply(inputs)


		with self.test_session(graph=current_graph) as test_session:

			test_session.run(tf.global_variables_initializer())

			# run
			outputs = test_session.run(outputs)

		self.assertIsNotNone(outputs)
		self.assertEqual(outputs.dtype, np.float32)
		self.assertEqual(outputs.shape, tuple(input_shape[:-1]))


