import tensorflow as tf
import numpy as np


class F0UnetGears:


	def __init__(self, hparams, level=None):

		self.hparams = hparams
		self.level = level


	def apply(self, inputs, training=True):

		return self.call(inputs, training=training)


	def call(self, inputs, training=True):
		"""
		This implements the U-Net architecture as described in
		[1] Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation
		For transposed convolution, see:
		[2] Dumoulin, V. and Visin, F., 2016. A guide to convolution arithmetic for deep learning
		-----
		VALID convolution : out = ceil(in-k+1/s)
		VALID transposed convolution :
				- out' = s(in'-1) + k
				- if (in-k) % s == 0, then in' = out = (in-k)/s + 1, so out' = in
		Args:
			inputs:
			training:
		Returns:
		"""
		hparams = self.hparams
		level = self.level

		debug_print = False

		floors_num_layers = hparams.unet_floors_num_layers
		total_num_floors = len(floors_num_layers)

		layers_kernel_shape = hparams.unet_layers_kernel_shape
		layers_pool_shape = hparams.unet_layers_pool_shape
		layers_pool_stride = hparams.unet_layers_pool_stride

		# if dims is a list of list, it means we have different values for each layer, otherwise we will use the same
		# duplicated values for each floor.
		if type(layers_kernel_shape[0]) is int:
			layers_kernel_shape = [layers_kernel_shape for _ in range(total_num_floors+1)] #+ 1 because we have a last conv at the bottom
		if type(layers_pool_shape[0]) is int:
			layers_pool_shape = [layers_pool_shape for _ in range(total_num_floors)]
		if type(layers_pool_stride[0]) is int:
			layers_pool_stride = [layers_pool_stride for _ in range(total_num_floors)]

		assert (len(layers_kernel_shape) - 1) == len(layers_pool_shape) == len(layers_pool_stride) == total_num_floors

		num_kernels = hparams.unet_first_floor_layers_num_kernels
		padding_type = 'SAME'

		outputs = inputs
		floors_down_sampled_outputs =[]


		# the down-sampling part of the U
		with tf.variable_scope('down_branch'):

			for count_floors, (floor_num_layers,
			                   floor_kernel_shape,
			                   floor_pool_shape,
			                   floor_pool_stride) in enumerate(zip(floors_num_layers,
			                                                       layers_kernel_shape[:-1],
			                                                       layers_pool_shape,
			                                                       layers_pool_stride)):

				with tf.variable_scope('floor_%d' % count_floors):

					for count_layers in range(floor_num_layers):

						with tf.variable_scope('layer_%d' % count_layers):

							# batch norm
							outputs = tf.layers.batch_normalization(outputs,
							                                        training=training)

							# if convolution is SAME, then we need to pad manually as we will need the padded output for the
							# up-sampling part
							if padding_type == 'SAME':
								outputs = self.pad(outputs, [floor_kernel_shape[0], floor_kernel_shape[1]])


							# 2 conv2D followed by Relu
							outputs = tf.layers.conv2d(inputs=outputs,
							                           filters=num_kernels,
							                           kernel_size=[floor_kernel_shape[0], floor_kernel_shape[1]],
							                           kernel_regularizer=None, # tf.contrib.layers.l2_regularizer(scale=regularizer_scale) if regularizer_scale > 0 else
							                           strides=1,
							                           padding='VALID', # if we have padded, then we can use valid
							                           activation=tf.nn.relu,
							                           use_bias=False,
							                           )

							#outputs = tf.layers.batch_normalization(outputs,
							#                                        training=training)
							#outputs = tf.nn.relu(outputs)

							if debug_print:
								outputs = tf.Print(outputs, [tf.shape(outputs)[1:]])

				# keep track of the output that will be reused in the ascending part of th U
				floors_down_sampled_outputs.append(outputs)

				with tf.variable_scope('down_scale_%d_to_%d' % (count_floors, count_floors + 1)):

					# max pool with stride. For SAME, it is a bit more complicated:
					#   - if in = 2d, then out = d, ok.
					#   - but if in = 2d+1, then either we don't pad, and out = d, or we pad and out = d+1, but in the
					#     up-sampling branch, we will have 2d or 2d+1 (none is 2d+1).
					# so we pad, and we have to make sure that we remove 1 dimension in the up-sampling branch
					# see (*) below

					outputs = tf.layers.average_pooling2d(outputs,
					                                      pool_size=[floor_pool_shape[0], floor_pool_shape[1]],
					                                      strides=[floor_pool_stride[0], floor_pool_stride[1]],
					                                      padding=padding_type,
					                                      )
					if debug_print:
						outputs = tf.Print(outputs, [tf.shape(outputs)[1:]])

				# we double the number of num_kernels at each floor
				num_kernels *= 2

			with tf.variable_scope('floor_%d' % total_num_floors):

				floor_kernel_shape = layers_kernel_shape[-1]

				# the bottom of the U, 2 conv2D with Relu
				for count_layers in range(2):

					with tf.variable_scope('layer_%d' % count_layers):

						# batch norm
						outputs = tf.layers.batch_normalization(outputs,
						                                        training=training)

						outputs = tf.layers.conv2d(inputs=outputs,
						                           filters=num_kernels,
						                           kernel_size=[floor_kernel_shape[0], floor_kernel_shape[1]],
						                           kernel_regularizer=None, # tf.contrib.layers.l2_regularizer(scale=regularizer_scale) if regularizer_scale > 0 else
						                           strides=1,
						                           padding=padding_type, # as specified in [1]
						                           activation=tf.nn.relu,
						                           use_bias=False,
						                           )

						#outputs = tf.layers.batch_normalization(outputs,
						#                                        training=training)
						#outputs = tf.nn.relu(outputs)

						if debug_print:
							outputs = tf.Print(outputs, [tf.shape(outputs)[1:]])


		with tf.variable_scope('up_branch'):

			with tf.variable_scope('floor_%d' % total_num_floors):
				outputs = tf.identity(outputs) # just to be consistent


			# caution: we do it in reverse order (starting bottom of the U)
			for count_floors, (floor_num_layers,
			                   floor_kernel_shape,
			                   floor_pool_shape,
			                   floor_pool_stride) in reversed(list(enumerate(zip(floors_num_layers,
			                                                                     layers_kernel_shape[:-1],
			                                                                     layers_pool_shape,
			                                                                     layers_pool_stride)))):

				# if we have reached the level, stop here the graph, and use current output as network output
				if level is not None and count_floors + 1 == level: #level == total_num_floors - count_floors:
					break

				with tf.variable_scope('upscale_%d_to_%d' % (count_floors+1, count_floors)):

					# batch norm
					outputs = tf.layers.batch_normalization(outputs,
					                                        training=training)

					# we divide by 2 the number of num_kernels at each floor
					num_kernels //= 2

					# 2x2 de-convolution (up-sampling reverse of the max pool)
					# down side was in ---> out, up side is in'= out ---> out'
					# condition to have out' = in is (in-k) % s
					outputs = tf.layers.conv2d_transpose(outputs,
					                                     filters=num_kernels,
					                                     kernel_size = [floor_pool_shape[0], floor_pool_shape[1]],
					                                     strides = [floor_pool_stride[0], floor_pool_stride[1]],
					                                     padding=padding_type,
					                                     activation=None)
					if debug_print:
						outputs = tf.Print(outputs, [tf.shape(outputs)[1:]])

				with tf.variable_scope('floor_%d' % count_floors):

					outputs_shape = tf.shape(outputs)

					down_sampled_outputs = floors_down_sampled_outputs[count_floors]
					down_sampled_outputs_shape = tf.shape(down_sampled_outputs)

					if padding_type == 'VALID':
						# here up link is always smaller that down link

						time_offset = (down_sampled_outputs_shape[1] - outputs_shape[1]) // 2
						frequency_offset =  (down_sampled_outputs_shape[2] - outputs_shape[2]) // 2
						down_sampled_outputs = down_sampled_outputs[:,time_offset:time_offset+outputs_shape[1], frequency_offset:frequency_offset+outputs_shape[2],:]

					else:
						# (*) here we might have an outputs with 1 extra bin for SAME
						outputs = outputs[:,:down_sampled_outputs_shape[1], :down_sampled_outputs_shape[2],:]

					# and concat with U up side along the depth dimension
					outputs = tf.concat([outputs, down_sampled_outputs], axis=-1)

					for count_layers in range(floor_num_layers):

						with tf.variable_scope('layer_%d' % count_layers):

							outputs = tf.layers.batch_normalization(outputs,
							                                        training=training)

							outputs = tf.layers.conv2d(inputs=outputs,
							                           filters=num_kernels,
							                           kernel_size=[floor_kernel_shape[0], floor_kernel_shape[1]],
							                           kernel_regularizer=None, # tf.contrib.layers.l2_regularizer(scale=regularizer_scale) if regularizer_scale > 0 else
							                           strides=1,
							                           padding=padding_type,
							                           activation=tf.nn.relu,
							                           use_bias=False,
							                           )

							#outputs = tf.layers.batch_normalization(outputs,
							#                                        training=training)
							#outputs = tf.nn.relu(outputs)

							if debug_print:
								outputs = tf.Print(outputs, [tf.shape(outputs)[1:]])


		with tf.variable_scope('squeeze'): # if level is None else 'squeeze_%d' % level):

			# batch norm
			outputs = tf.layers.batch_normalization(outputs,
			                                        training=training)
			# last 1x1 convolution
			outputs = tf.layers.conv2d(inputs=outputs,
			                           filters=1,
			                           kernel_size=1,
			                           kernel_regularizer=None,
			                           strides=1,
			                           padding='SAME',
			                           activation=tf.nn.sigmoid, # CAUTION: we sigmoid here, so we output a probability, do not use sigmoid cross entropy for loss
			                           use_bias=False,
			                           )

			outputs = tf.squeeze(outputs, -1)  # [b,t,f]


		return outputs





	def pad(self, inputs, kernel_shape, stride_shape = None):
		"""
		Pads the input along the spatial dimensions independently of input size.
		This ensures that the convolution of the input by the kernel will have same size than the input.
		Args:
		    inputs: A tensor of size [batch, height_in, width_in, channels].
		    kernel_shape: The kernel to be used in the conv2d or max_pool2d operation.
					   Should be a positive integer
			stride_shape:
		Returns:
		  A tensor with the same format as the input with the data either intact
		  (if kernel_size == 1) or padded (if kernel_size > 1).
		"""
		hparams = self.hparams
		#padding_mode = hparams.unet_layers_padding_mode

		kernel_shape = np.array(kernel_shape, dtype=np.int32)

		if stride_shape is None:

			pad_total = kernel_shape - 1

		else:

			stride_shape = np.array(stride_shape, dtype=np.int32)
			inputs_shape = tf.shape(inputs)
			inputs_shape = inputs_shape[1:2]

			pad_total = tf.cond(tf.equal(tf.mod(inputs_shape, stride_shape), 0),
			                    lambda : kernel_shape - stride_shape,
			                    lambda : kernel_shape - tf.mod(inputs_shape, stride_shape))

		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg

		padded_inputs = tf.pad(inputs,
		                       paddings=[[0, 0], [pad_beg[0], pad_end[0]],
		                                 [pad_beg[1], pad_end[1]], [0, 0]]) #mode=padding_mode)

		return padded_inputs


	@staticmethod
	def unet_output_shape(hparams, input_shape):
		"""
		Utility function that computes the U-net output_shape for each layer.
		Reminder: https://www.tensorflow.org/api_guides/python/nn#Convolution
			VALID convolution: out = ceil[(in - k + 1) / s]
			VALID transposed convolution:
				- out' = s(in'-1) + k
				- if (in-k) % s == 0, then in' = out = (in-k)/s + 1, so out' = in
		Args:
			hparams:
			input_shape: tuple, of the form [t, f, d]
		Returns:
			output_shape: tuple, of the form [t, f, d]
		"""

		floors_num_layers = hparams.unet_floors_num_layers
		total_num_floors = len(floors_num_layers)

		layers_kernel_shape = hparams.unet_layers_kernel_shape
		layers_pool_shape = hparams.unet_layers_pool_shape
		layers_pool_stride = hparams.unet_layers_pool_stride


		# if dims is a list of list, it means we have different values for each layer, otherwise we will use the same
		# duplicated values for each floor.
		if type(layers_kernel_shape[0]) is int:
			layers_kernel_shape = [layers_kernel_shape for _ in range(total_num_floors+1)]
		if type(layers_pool_shape[0]) is int:
			layers_pool_shape = [layers_pool_shape for _ in range(total_num_floors)]
		if type(layers_pool_stride[0]) is int:
			layers_pool_stride = [layers_pool_stride for _ in range(total_num_floors)]

		assert (len(layers_kernel_shape) - 1) == len(layers_pool_shape) == len(layers_pool_stride) == total_num_floors


		num_kernels = hparams.unet_first_floor_layers_num_kernels
		padding_type = hparams.database_data_chunks_padding_type.name

		output_shapes = [input_shape]

		output_shape = np.array(input_shape)
		output_depth = output_shape[-1]
		output_shape = output_shape[:-1]

		# down
		for floor_num_layers, \
		    floor_kernel_shape, \
		    floor_pool_shape, \
		    floor_pool_stride in zip(floors_num_layers,
		                             layers_kernel_shape[:-1],
		                             layers_pool_shape,
		                             layers_pool_stride):

			kernel_shape = np.array(floor_kernel_shape)
			pool_shape = np.array(floor_pool_shape)
			pool_stride_shape = np.array(floor_pool_stride)

			for _ in range(floor_num_layers):

				# 2D convolution followed by Relu
				output_shape = output_shape if padding_type == 'SAME' else np.ceil(np.float32(output_shape - kernel_shape + 1) / 1)
				output_depth = num_kernels

				# store
				output_shapes.append(list(output_shape) + [output_depth])

			# 2x2 max pool wit 2 stride
			output_shape = np.ceil(np.float32(output_shape - pool_shape + 1) / np.float32(pool_stride_shape))

			# store
			output_shapes.append(list(output_shape) + [output_depth])

			# we double the number of num_kernels at each floor
			num_kernels *= 2

		# bottom
		for _ in range(2):

			floor_kernel_shape = layers_kernel_shape[-1]

			# 2D convolution followed by Relu
			output_shape = output_shape if padding_type == 'SAME' else np.ceil(np.float32(output_shape - floor_kernel_shape + 1) / 1)
			output_depth = num_kernels
			# store
			output_shapes.append(list(output_shape) + [output_depth])

		# up
		for floor_num_layers, \
		    floor_kernel_shape, \
		    floor_pool_shape, \
		    floor_pool_stride in reversed(list(zip(floors_num_layers,
		                                           layers_kernel_shape[:-1],
		                                           layers_pool_shape,
		                                           layers_pool_stride))):

			kernel_shape = np.array(floor_kernel_shape)
			pool_shape = np.array(floor_pool_shape)
			pool_stride_shape = np.array(floor_pool_stride)

			# we divide by 2 the number of num_kernels at each floor
			num_kernels //= 2

			# 2x2 transposed convolution with 2 stride
			output_shape = pool_stride_shape*(output_shape-1) + pool_shape
			output_depth = num_kernels
			# store
			output_shapes.append(list(output_shape) + [output_depth*2]) # * 2 because we concatenate with down branch

			for _ in range(floor_num_layers):

				# 2D convolution followed by Relu
				output_shape = output_shape if padding_type == 'SAME' else np.ceil(np.float32(output_shape - kernel_shape + 1) / 1)
				# store
				output_shapes.append(list(output_shape) + [output_depth])

		# last
		output_shapes.append(list(output_shape) + [1])

		return output_shapes