


def get_database_chunks_shape_in_bins(hparams):

	chunks_duration_in_bins = int(hparams.database_data_chunks_duration_in_sec*hparams.corpus_data_sec_to_bins)

	if type(hparams.database_data_chunks_overlap_in_sec) is list:
		chunks_beg_overlap_in_bins = int(hparams.database_data_chunks_overlap_in_sec[0]*hparams.corpus_data_sec_to_bins)
		chunks_end_overlap_in_bins = int(hparams.database_data_chunks_overlap_in_sec[1]*hparams.corpus_data_sec_to_bins)
	else:
		chunks_beg_overlap_in_bins = 0
		chunks_end_overlap_in_bins = int(hparams.database_data_chunks_overlap_in_sec*hparams.corpus_data_sec_to_bins)

	return chunks_duration_in_bins, (chunks_beg_overlap_in_bins, chunks_end_overlap_in_bins)




def chunk_data_with_same_padding(data,
                                 data_chunks_duration_in_bins,
                                 data_chunks_overlap_in_bins,
                                 ):
	"""
	Chunks data.
	Args:
		data: ndarray, [t, ...]
		data_chunks_duration_in_bins:
		data_chunks_overlap_in_bins:
	Returns:
	"""


	if type(data_chunks_overlap_in_bins) is list or type(data_chunks_overlap_in_bins) is tuple:
		chunks_beg_overlap_in_bins = data_chunks_overlap_in_bins[0]
		chunks_end_overlap_in_bins = data_chunks_overlap_in_bins[1]
	else:
		chunks_beg_overlap_in_bins = 0
		chunks_end_overlap_in_bins = data_chunks_overlap_in_bins

	start_bin = 0
	end_bin = start_bin + data_chunks_duration_in_bins
	chunks = []

	while end_bin < data.shape[0]:

		chunks.append(data[start_bin:end_bin])

		start_bin = end_bin - (chunks_beg_overlap_in_bins + chunks_end_overlap_in_bins)
		end_bin = start_bin + data_chunks_duration_in_bins

	# save last chunk
	end_bin = data.shape[0]
	if end_bin > start_bin:
		last_chunk = data[start_bin:end_bin]

	else:
		last_chunk = None

	return chunks, last_chunk