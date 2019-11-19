import tensorflow as tf
import numpy as np
import os

from f0estimator.model.f0_unet import F0Unet

import f0estimator.helpers.chunker as chunker
import f0estimator.helpers.utils as utils

F0_LOGGER_NAME = 'f0_estimator'
F0_MODEL_DIRECTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'f0')


class F0ModelRunner:

	def __init__(self,
	             model_directory_path=None,
	             devices = None
	             ):
		"""
		Args:
			model_directory_path:
			devices:
		"""
		if model_directory_path is None:
			model_directory_path = F0_MODEL_DIRECTORY_PATH

		if not os.path.exists(model_directory_path):
			raise ValueError('There is no model at %s.' % model_directory_path)

		if not os.path.exists(os.path.join(model_directory_path, 'checkpoint')):
			raise ValueError('Model directory %s has no checkpoint file.' % model_directory_path)

		if devices is None:
			devices = utils.acquire_devices()

		# make all this global
		self.hparams = None # those are the generic model hparams
		self.devices = devices # the devices managed by this runner

		self.mode = tf.estimator.ModeKeys.PREDICT
		self.model_directory_path = model_directory_path

		# to be filled by concrete subclass
		self.model_class = F0Unet
		self.model_class_kwargs = None
		self.model = None

		self.logger = utils.get_logger(F0_LOGGER_NAME)



	def build_model(self, **kwargs):

		if not hasattr(self, 'model_class') or self.model_class is None:
			raise ValueError('A model class must have been set before model can be instantiated.')

		self.model_class_kwargs = kwargs if kwargs else {} # this is to reuse for eval model

		model_directory_path = self.model_directory_path
		model_class = self.model_class

		model = model_class(model_directory_path=model_directory_path,
		                    **kwargs)

		# get devices
		model.build(devices=self.devices)

		self.model = model




	def apply_model(self, sources_filepaths, save_directory_path=None):
		"""
		Applies the trained model to the data stored at source_filepath.
		It divides the data into chunks of the length that was used for model training, process the chunks, and
		concatenate them back.
		Args:
			sources_filepaths:
			save_directory_path:
		Returns:
		"""
		if not hasattr(self, 'model') or self.model is None:
			raise ValueError('A model must exist before it can be applied.')

		model = self.model
		if model.mode is not tf.estimator.ModeKeys.PREDICT:
			raise ValueError("The model instance must have been created with PREDICT mode (found %s)." % model.mode)

		if save_directory_path is None:
			raise ValueError("Please provide a directory to save the results.")

		if not os.path.exists(save_directory_path):
			os.mkdir(save_directory_path)

		save_files_extension = 'f0.npy'

		target_filepaths = []
		for i, source_filepath in enumerate(sources_filepaths):

			# load the data
			try:

				if 'hcqt.npy.gz' in source_filepath:
					tgt_filename = '.'.join(os.path.basename(source_filepath).split('.')[:-3]) # assumes name.xxx.yy.hcqt.npy.gz
				else:
					tgt_filename = '.'.join(os.path.basename(source_filepath).split('.')[:-2]) # assumes name.xxx.yy.hcqt.npy

				tgt_filepath = os.path.join(save_directory_path, '%s.%s' % (tgt_filename, save_files_extension))

				if not os.path.exists(tgt_filepath):

					src_data = utils.load_data(source_filepath) # [f, t] or [h, f, t]
					src_data = np.transpose(src_data) # [t, ...]

					tgt_data = self.apply_for_same_padding(src_data)
					# saver
					utils.save_data(tgt_filepath, tgt_data)

				target_filepaths.append(tgt_filepath)

			except (EOFError, OSError):
				self.logger.warning('Could not load %s...' % source_filepath)



		return target_filepaths






	def apply_for_same_padding(self, src_data):

		model = self.model
		hparams = model.hparams

		data_chunks_duration_in_bins, \
		data_chunks_overlap_in_bins = chunker.get_database_chunks_shape_in_bins(hparams)

		data_chunks_beg_overlap_in_bins = data_chunks_overlap_in_bins[0]
		data_chunks_end_overlap_in_bins = data_chunks_overlap_in_bins[1]

		chunks, \
		last_chunk = chunker.chunk_data_with_same_padding(src_data,
		                                                  data_chunks_duration_in_bins=data_chunks_duration_in_bins,
		                                                  data_chunks_overlap_in_bins=data_chunks_overlap_in_bins) # [t, ...]

		# pile all the chunks (the number of chunk will be the batch size).
		src_data_chunks = np.stack(chunks)  #[num_chunks, t, ...]

		# now apply network
		num_chunks = src_data_chunks.shape[0]
		batch_size = hparams.dataset_eval_batch_size

		outputs = []

		for i in range(0, num_chunks, batch_size):

			if i + batch_size < num_chunks:
				dataset_placeholders = [src_data_chunks[i:i+batch_size]] # [num_batch, t, ...]
			else:
				dataset_placeholders = [src_data_chunks[i:]]

			batched_output = model.apply(dataset_placeholders=dataset_placeholders) # [num_batch, t]

			# we might have overlapped, so we cut that part, except for the very first one
			if i == 0 and data_chunks_beg_overlap_in_bins > 0:
				outputs.append(batched_output[0, :data_chunks_beg_overlap_in_bins])

			# all the other chunks are trimmed at the beginning and the end
			if data_chunks_beg_overlap_in_bins > 0:
				batched_output = batched_output[:, data_chunks_beg_overlap_in_bins:]
			if data_chunks_end_overlap_in_bins > 0:
				batched_output = batched_output[:, :-data_chunks_end_overlap_in_bins]

			# concatenate along time axis
			batched_output = np.concatenate(batched_output.tolist(), axis=0) # [num_batch*t]
			batched_output = batched_output.astype(np.float32) # important: concatenate changes the dtype !!!
			outputs.append(batched_output)

		# add the last chunk
		if last_chunk is not None:
			last_data = np.expand_dims(last_chunk, 0) # [b=1, t, ...]
			dataset_placeholders = [last_data]
			batched_output = model.apply(dataset_placeholders=dataset_placeholders)  #[b=1, t, ...]

			last_output = np.squeeze(batched_output, 0) #[t,...]

			# remove the beginning
			if data_chunks_beg_overlap_in_bins > 0:
				last_output = last_output[data_chunks_beg_overlap_in_bins:]
			outputs.append(last_output)

		# concatenate all
		output = np.concatenate(outputs, axis=0) # [t,...]
		output = output.astype(np.float32) # important, concatenate changes the dtype !!!

		return output