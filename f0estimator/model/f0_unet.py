import tensorflow as tf
import os
import json
import collections
import enum


from f0estimator.model.f0_unet_gears import F0UnetGears

import f0estimator.helpers.utils as utils
from f0estimator.helpers.utils import SimpleHParams


class F0Database:

	class DataType(enum.Enum):
		HCQT_TO_F0 = 0
		HCQT_TO_MULTI_F0 = 1

	class DataPaddingType(enum.Enum):
		SAME = 0
		#VALID = 1



class F0Dataset(
	collections.namedtuple("MelodyEstimatorDataset",
	                       ('initializer',
	                        'placeholders',
	                        'batch_size',
	                        'batched_source',
	                        #'batched_target',
	                        ))):
	pass



class F0Unet:
	"""
	This model extracts f0 out of the CQT.
	It is trained with chunks of CQT extracted from data-augmented MDB corpus compared to corresponding ground truth f0.
	"""

	class InitializerType(enum.Enum):
		UNIFORM = 0
		GLOROT_NORMAL = 1
		GLOROT_UNIFORM = 2
		XAVIER_NORMAL = 3
		XAVIER_UNIFORM = 4



	def __init__(self,
	             model_directory_path=None,
	             **kwargs
	             ):
		"""
		Creates a model in PREDICT mode: reads its config, builds graph and loads its latest checkpoint.

		Args:
			model_directory_path:
		"""


		# if model directory does not exist, it means this is a first training, so we create a directory
		# otherwise, it is a subsequent training
		if model_directory_path is None or not os.path.exists(model_directory_path):
			raise RuntimeError('The model %s does not exist' % model_directory_path)

		self.directory_path = model_directory_path
		self.mode = tf.estimator.ModeKeys.PREDICT

		# config
		self.config_directory_path = model_directory_path # by default, it is the model_directory_path
		self.hparams = self.read_config_as_hparams(os.path.join(model_directory_path, 'config.txt'))

		# build
		self.graph = tf.Graph()
		self.built = False
		self.reuse = None

		# load
		self.loader = None # this will be set when building
		self.load_checkpoints_directory_path = model_directory_path # by default, it is the model_directory_path
		self.loaded = False
		self.reloaded = False # this is to give information whether the model was loaded the first time, or reloaded

		# save
		self.saver = None # this will be set when building
		self.save_checkpoints_directory_path = model_directory_path # by default, it is the model_directory_path

		# misc
		self.tag = 'f0_unet'

		# for future use
		self.devices = None
		self.session = None
		self.dataset = None
		self.loss = None
		self.optimizer = None
		self.learning_rate = None
		self.global_step = None
		self.output = None

		self.model_gears = None


	@staticmethod
	def read_config_as_hparams(config_filepath):

		if not os.path.exists(config_filepath):
			raise ValueError('There is no config file at %s.' % config_filepath)

		config = utils.read_config(config_filepath)

		# matcher config
		hparams = SimpleHParams(
			dict(

				# unet
				unet_floors_num_layers = json.loads(config.get('unet', 'unet_floors_num_layers')),
				unet_first_floor_layers_num_kernels = config.getint('unet', 'unet_first_floor_layers_num_kernels'),
				unet_layers_kernel_shape = json.loads(config.get('unet', 'unet_layers_kernel_shape')),
				unet_layers_pool_shape = json.loads(config.get('unet', 'unet_layers_pool_shape')),
				unet_layers_pool_stride = json.loads(config.get('unet', 'unet_layers_pool_stride')),
				unet_training_level_by_level = config.getboolean('unet', 'unet_training_level_by_level', fallback=False),

				# database
				database_data_type = F0Database.DataType[config.get('database', 'database_data_type')],
				database_data_chunks_duration_in_sec = config.getfloat('database', 'database_data_chunks_duration_in_sec'),
				database_data_chunks_overlap_in_sec = json.loads(config.get('database', 'database_data_chunks_overlap_in_sec')),
				database_data_chunks_padding_type = F0Database.DataPaddingType[config.get('database', 'database_data_chunks_padding_type')],

				database_src_data_shape = json.loads(config.get('database', 'database_src_data_shape')),
				database_tgt_data_shape = json.loads(config.get('database', 'database_tgt_data_shape')),

				# dataset
				dataset_eval_batch_size = config.getint('dataset', 'dataset_eval_batch_size'),

				# audio
				corpus_data_sec_to_bins = config.getfloat('corpus', 'corpus_data_sec_to_bins')

			)
		)

		return hparams


	def build(self, devices=None):
		"""
		Builds the model based on input shapes received.
		This is to be used for subclassed models, which do not know at instantiation time what their inputs look like.
		Args:
		"""
		# get the devices
		if self.built:
			return

		hparams = self.hparams
		mode = self.mode
		graph = self.graph

		# lock the devices that we will need in the graph
		self.devices = devices if devices else ['/cpu:0']

		with graph.as_default(), tf.container(mode):

			#self.global_step = tf.Variable(0, trainable=False, name='global_step')
			#self.learning_rate = tf.Variable(hparams.training_learning_rate, trainable=False, name='learning_rate')

			self.build_graph() # <---- put the concrete subclass logic here

			# add a load and a saver
			self.saver = tf.train.Saver(tf.global_variables(), name='saver')
			self.loader = self.saver # by default we load and save same variables.


		# assert that the network output something
		if not hasattr(self, 'output') or self.output is None:
			raise AssertionError('The model must have built an output.')

		# done
		self.built = True




	def build_dataset(self):
		"""
		This is usually called within build_graph, but is provided here as convenience.
		Builds the dataset, depending whether or not a database has been provided.
		If so, the model will be used in training mode, and the database will be used to create a TfRecordsDataset out
		of the tfrecords files of the database.
		If not, the model will be used in apply mode, and a TensorDataset will be created to be feed directly when needed
		via placeholders.
		Returns:
			dataset: a Dataset object.
		"""

		hparams = self.hparams

		with tf.variable_scope('dataset'):

			placeholder_shape = [None, None] + hparams.database_src_data_shape[1:] # [b, t, ...] we put None for batch as it is unknown, and None for time to handle variable lengths chunks

			# placeholders
			source_data_placeholder = tf.placeholder(shape=placeholder_shape, dtype=tf.float32)

			# create one dataset object out of the list of tfrecords files and put the cursor at skip count
			dataset = tf.data.Dataset.from_tensors(source_data_placeholder)

			dataset = dataset.map(lambda src_data: (src_data,)) # be careful, the comma counts as we must pass a list!

			dataset_iterator = dataset.make_initializable_iterator()


			(src_data,
			 ) = dataset_iterator.get_next()

			dataset = F0Dataset(
				initializer=dataset_iterator.initializer,
				placeholders = [source_data_placeholder],
				batch_size = tf.shape(src_data)[0],
				batched_source = src_data,
			)

		self.dataset = dataset

		return dataset





	def build_graph(self):

		hparams = self.hparams
		mode = self.mode
		devices = self.devices

		dataset = self.build_dataset()

		with tf.device(devices[0]):

			with tf.variable_scope(self.tag):

				# create an initializer
				#initializer = self.build_initializer()
				#tf.get_variable_scope().set_initializer(initializer)

				model_gears = F0UnetGears(hparams)

			inputs = dataset.batched_source # [b,t,f,h]

			# only inference here
			if mode == tf.estimator.ModeKeys.PREDICT:

				outputs = model_gears.apply(inputs,
				                            training= mode == tf.estimator.ModeKeys.TRAIN) # [b, t, f]

			else:
				raise ValueError('Mode %s is not supported.' % mode)


		# custom for this model in particular
		self.model_gears = model_gears

		# finally keep track of the tensors we will need to execute this model
		self.output = outputs





	def build_initializer(self):

		hparams = self.hparams
		seed = None

		if hparams.training_initializer_type == F0Unet.InitializerType.UNIFORM:
			return tf.random_uniform_initializer(-hparams.training_initializer_weight, hparams.training_initializer_weight, seed=seed)
		elif hparams.training_initializer_type == F0Unet.InitializerType.GLOROT_NORMAL:
			return tf.keras.initializers.glorot_normal(seed=seed)
		elif hparams.training_initializer_type == F0Unet.InitializerType.GLOROT_UNIFORM:
			return tf.keras.initializers.glorot_uniform(seed=seed)
		elif hparams.training_initializer_type == F0Unet.InitializerType.XAVIER_NORMAL:
			return tf.contrib.layers.xavier_initializer(uniform=False, seed=seed)
		elif hparams.training_initializer_type == F0Unet.InitializerType.XAVIER_UNIFORM:
			return tf.contrib.layers.xavier_initializer(uniform=True, seed=seed)
		else:
			raise ValueError("Unknown initializer type %s" % hparams.training_initializer_type)



	def apply(self, dataset_placeholders = None):
		"""
		This applies the trained network to some data.
		Args:
			dataset_placeholders:
		Returns:
		"""

		if not self.built:
			raise ValueError('The model must have been built before you can apply it.')

		if not self.loaded:
			# we call init_or_load once so that apply can be called several time without reloading the model
			self.init_or_reload()

		session = self.session

		# initialize the dataset
		dataset_initializer_feed_dict =  {self.dataset.placeholders[i]: p for i, p in enumerate(dataset_placeholders)}
		session.run(self.dataset.initializer,
		            feed_dict=dataset_initializer_feed_dict)

		# run a complete epoch on dataset
		epoch_done = False
		output = None

		while not epoch_done:

			try:

				# run one batch
				output = session.run(self.output)

			except tf.errors.OutOfRangeError:
				epoch_done = True # we go through the infer dataset only once


		return output



	def init_or_reload(self, session=None):
		"""
		Starts a session if needed, and initialize the model or reload it if it was previously saved.
		Args:
			session: Session object, the session to use.
		Returns:
		"""

		if not self.built:
			raise ValueError('The model must have been built before you can reload it.')

		#hparams = self.hparams
		graph = self.graph

		checkpoints_directory_path = self.load_checkpoints_directory_path
		loader = self.loader

		if not self.loaded:
			# we don't have a session yet
			if session is None:
				device_config = utils.build_device_config()
				session = tf.Session(target='', graph=graph, config=device_config)

			self.session = session # keep reference

		else:
			assert self.session is not None, "Loaded model should have a session already."

		# we always reload from the training saved model (if it was saved)
		latest_training_checkpoint = tf.train.latest_checkpoint(checkpoints_directory_path)

		with self.graph.as_default():

			if latest_training_checkpoint is None:

				self.session.run(tf.global_variables_initializer()) # all variables that will be used must initialized before being evaluated

				# this is the first time we load this model, so it has no session yet
				self.reloaded = False

			else:

				# there was a checkpoint, so restore the model from there
				loader.restore(self.session, latest_training_checkpoint)
				#self.init_uninitialized_variables() # hook in case some variables exist in current model that were not reloaded from checkpoint
				self.reloaded = True

		#self.print_num_parameters(True)

		self.loaded = True








