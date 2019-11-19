import tensorflow as tf
import os
import numpy as np
import gzip
import time
import logging, logging.handlers, sys
from configparser import ConfigParser


def acquire_devices(num_devices=1):

	TITAN_XP_ID = 2
	CPU = '/cpu'
	GPU = '/gpu'

	try:
		import manage_gpus as mgp

		gpu_ids = mgp.board_ids()

		if gpu_ids is not None:

			devices = []

			# we have one or more GPU, see if the Titan is available, and if so, take it
			if TITAN_XP_ID in gpu_ids:

				locked_gpu_id = mgp.obtain_lock_id(id=TITAN_XP_ID)

				if locked_gpu_id < 0:
					raise RuntimeError("Can not obtain lock for GPU %d" % TITAN_XP_ID)
				devices.append('%s:%d' % (GPU, locked_gpu_id))

				gpu_ids.remove(locked_gpu_id)

			for i in range(num_devices - len(devices)):

				locked_gpu_id = mgp.obtain_lock_id(id=-1) # -1: next available

				if locked_gpu_id >= 0:
					devices.append('%s:%d' % (GPU, locked_gpu_id))

			if len(devices) == 0:
				raise RuntimeError("Can not obtain lock for any GPU.")

		else:
			devices = ['%s:0' % CPU]

	except ImportError:

		# we're apparently not where we usually are, so get whatever is there
		devices = ['%s:0' % GPU] if tf.test.is_gpu_available() else ['%s:0' % CPU]

	except AttributeError: # this might happens with old versions of manage_gpus
		devices = ['%s:0' % CPU]


	return devices


def build_device_config():
	"""
	Set up GPU options.
	Args:
		hparams:
	Returns:
		-
	"""

	config_proto = tf.ConfigProto(
		allow_soft_placement=True, # do we allow TF to switch to another devices if ours is not available ?
		log_device_placement=False, # do we log the devices usage ?
	)
	config_proto.gpu_options.allow_growth = True
	#config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9

	return config_proto

def save_data(filepath, data):
	"""
		A convenience function to deal transparently with both compressed and un-compressed data.
	"""
	#np.savez_compressed(filepath, data) if filepath.endswith('npz') else np.save(filepath, data)
	if filepath.endswith('gz'):
		f = gzip.GzipFile(filepath, "w")
		np.save(f, data)
		f.close()
	else:
		np.save(filepath, data)



def load_data(filepath, mmap_mode=None):
	"""
		A convenience function to deal transparently with both compressed and un-compressed data.
	"""
	#data = np.load(filepath)
	#if filepath.endswith('npz'):
	#	data = data['arr_0']

	if filepath.endswith('gz'):
		f = gzip.GzipFile(filepath, "r")
		data = np.load(f)
		f.close()
	else:
		data = np.load(filepath, mmap_mode=mmap_mode)

	return data


def get_logger(name=None, level=logging.INFO, log_to_console=True, logs_directory_path=None, logs_filename='log.txt'):

	if name is None:
		name = __name__

	log = logging.getLogger(name)

	if not len(log.handlers):
		# if this logger has never been configured, then we add the handlers

		# configure a new logger
		log.setLevel(level)

		if log_to_console:
			handler = logging.StreamHandler(sys.stdout)
			handler.setLevel(level)
			log.addHandler(handler)

		if logs_directory_path:
			# rotating
			max_bytes = 10*1024*1024 # 10 MB
			backup_count = 2
			handler = logging.handlers.RotatingFileHandler(os.path.join(logs_directory_path, logs_filename),
			                                               maxBytes=max_bytes,
			                                               backupCount=backup_count)
			#handler = logging.FileHandler(os.path.join(logs_directory_path, logs_filename))
			handler.setLevel(level)
			log.addHandler(handler)

	return log


def elapsed_since(last_time):
	"""
	Convenience function that returns a formatted string giving the time elapsed since last_time.
	"""
	return time.strftime('%H:%M:%S', time.gmtime(time.time() - last_time))



def elapsed_per_epoch_since(last_time, num_epoch):
	"""
	Convenience function that returns a formatted string giving the time elapsed since last_time.
	"""
	return time.strftime('%H:%M:%S', (time.gmtime((time.time() - last_time )/num_epoch)))


def eta_based_on_elapsed_time(done, total, start_time):

	seconds_to_go = int((time.time() - start_time)*(total-done)/done)
	days_to_go = seconds_to_go // (3600*24)

	if days_to_go > 0:
		seconds_to_go = seconds_to_go - days_to_go * 3600*24
		time_to_go = '%s:%s' % (days_to_go, time.strftime('%H:%M:%S', time.gmtime(seconds_to_go)))

	else:
		time_to_go = time.strftime('%H:%M:%S', time.gmtime(seconds_to_go))

	return time_to_go



def read_config(config_filepath):
	"""
	Generates the configuration object to use for default directory paths.
	Args:
		config_filepath: the file to be loaded.
	Returns:
		config: configparser, the object to use for default directory paths.
	"""
	config = ConfigParser()
	config.read(config_filepath)

	return config





class SimpleHParams(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.__getitem__ #dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def add(self, a_dict):
		return self.update(a_dict)

	def set_if_not_none(self, key, value):
		if value is not None:
			dict.__setitem__(self, key, value)


	# for pickle/unpickle
	def __getstate__(self):
		return self.__dict__

	def __setstate__(self, d):
		return self.__dict__.update(d)


	# print
	def __repr__(self):
		return '\n'.join(['%s: %s' % (k, v) for k, v in self.items()])