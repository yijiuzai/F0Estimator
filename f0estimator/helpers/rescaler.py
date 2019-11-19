import numpy as np
import skimage.transform as skt


def rescale_data(data,
                 data_sec_to_bins,
                 data_num_octaves,
                 data_num_bins_per_octave,
                 data_new_range_in_octaves=3,
                 data_down_scale_factor=5.0,
                 data_new_offset_in_sec=0.0,
                 data_new_duration_in_sec=180,
                 data_new_duration_in_bins=1024,
                 data_zero_threshold=0.0,
                 data_normalize=False):
	"""
	Rescale data.
	It expects an array of shape [t, f] or [t, f, h].

	Args:
		data:
		data_sec_to_bins: float, the current sec_to_bins ratio
		data_num_octaves: int, the current number of octaves of the data
		data_num_bins_per_octave: int, the current number of bins per octave of the data
		data_new_range_in_octaves:
		data_down_scale_factor:
		data_new_offset_in_sec:
		data_new_duration_in_sec:
		data_new_duration_in_bins:
		data_zero_threshold:
		data_normalize:
	Returns:
	"""
	data_shape = data.shape
	if data_shape[1] != data_num_octaves * data_num_bins_per_octave:
		raise AssertionError('Data does not seem to be time major (found shape (%d, %d))' % data_shape)

	# threshold
	if data_zero_threshold > 0:
		data[data < data_zero_threshold] = 0.0

	# crop in frequency and center along the mean frequency bin
	if data_new_range_in_octaves <= 0:
		data_new_range_in_octaves = data_num_octaves
	if data_new_range_in_octaves < data_num_octaves:

		frequency_bins = np.argmax(data, axis=1)
		frequency_bins = frequency_bins[frequency_bins>0] # we remove the empty time indices

		mean_frequency_bin = int(np.mean(frequency_bins))

		data_frequency_range_in_bins = data_new_range_in_octaves*data_num_bins_per_octave
		data_frequency_half_lower_range_in_bins = data_frequency_range_in_bins // 2
		data_frequency_half_upper_range_in_bins = data_frequency_range_in_bins-data_frequency_half_lower_range_in_bins

		if mean_frequency_bin - data_frequency_half_lower_range_in_bins < 0:
			# we pad data up to the missing bins
			data = np.pad(data, [[0,0], [data_frequency_half_lower_range_in_bins-mean_frequency_bin,0]], mode='constant')
			data = data[:, :data_frequency_range_in_bins]
		elif mean_frequency_bin+data_frequency_half_upper_range_in_bins > data.shape[1]:
			data = np.pad(data, [[0,0], [0,mean_frequency_bin+data_frequency_half_upper_range_in_bins-data.shape[1]]], mode='constant')
			data = data[:, -data_frequency_range_in_bins:]
		else:
			data = data[:, mean_frequency_bin-data_frequency_half_lower_range_in_bins:mean_frequency_bin+data_frequency_half_upper_range_in_bins]

	sec_to_bins = data_sec_to_bins


	# crop in time
	data_offset_in_bins = int(data_new_offset_in_sec * sec_to_bins) if data_new_offset_in_sec > 0 else 0

	if data_new_duration_in_sec and data_new_duration_in_sec > 0:

		data_duration_in_bins = int(data_new_duration_in_sec * sec_to_bins)

		actual_data_duration_in_bins = data.shape[0]

		# we crop in time starting at offset. If song is not long enough, we keep it until the end.
		if actual_data_duration_in_bins > data_offset_in_bins + data_duration_in_bins:
			data = data[data_offset_in_bins:data_offset_in_bins+data_duration_in_bins]

		else:
			data = data[data_offset_in_bins:]
	# NOTE: we don't pad, as we don't want that the zero padding is interpreted as coverness !


	# downscale (this keeps the same time/frequency ratio)
	if data_down_scale_factor > 1.0:
		time_range_in_bins = data.shape[0]
		freq_range_in_bins = data.shape[1]

		if data.ndim == 2:
			data = skt.rescale(data, scale = 1/data_down_scale_factor, mode='reflect', multichannel=False) # [t, f]
			data = data[:int(time_range_in_bins/data_down_scale_factor), :int(freq_range_in_bins/data_down_scale_factor)] # this forces to a known dimensions as the rescale might be 1 pixel wider than this value

		else:
			# we are having an HCQT
			rescaled_cqts = [skt.rescale(data[:,:,i], scale = 1/data_down_scale_factor, mode='reflect', multichannel=False) for i in range(data.shape[-1])]
			rescaled_cqts = [rescaled_cqt[:int(time_range_in_bins/data_down_scale_factor), :int(freq_range_in_bins/data_down_scale_factor)] for rescaled_cqt in rescaled_cqts]
			data = np.stack(rescaled_cqts, axis=2) # [t, f, h]

		sec_to_bins /= data_down_scale_factor



	# rescale time-wise to get all songs to the same duration. This is equivalent to a tempo change.
	if data_new_duration_in_bins and data_new_duration_in_bins > 0:

		if data.ndim == 2:
			data = skt.resize(data, output_shape=[data_new_duration_in_bins, data.shape[1]]) # [t, f]

		else:
			resized_cqts = [skt.resize(data[:,:,i], output_shape=[data_new_duration_in_bins, data.shape[1]]) for i in range(data.shape[-1])]
			data = np.stack(resized_cqts, axis=2) # [t, f, h]

	# else
	# just make sure that we have a decent resize to make, otherwise simply trim
	#expected_duration_in_bins = int(data_new_duration_in_sec * sec_to_bins)
	#if data.shape[0] > expected_duration_in_bins:
	#	data = data[:expected_duration_in_bins]
	#elif data.shape[0] < expected_duration_in_bins:
	#	data = np.pad(data, [[0, data.shape[0] - expected_duration_in_bins], [0,0]], mode='constant')


	# threshold again in case interpolation 2d has created artifacts
	if data_zero_threshold > 0:
		data[data < data_zero_threshold] = 0.0

	if data_normalize:
		# re-norm along the frequency axis
		max_frequencies = np.max(data, axis=1, keepdims=True) #[t, f, ...]
		max_frequencies[max_frequencies<0.01] = 1.0 # avoid div zero
		data /= max_frequencies


	data = data.astype(np.float32)

	return data