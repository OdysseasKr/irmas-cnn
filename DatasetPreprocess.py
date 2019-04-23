from __future__ import print_function
import librosa
from pydub import AudioSegment
import numpy as np
import h5py
import sys
import os

class DatasetPreprocessor:
	""" Preprocesses dataset and creates h5 files """
	mode = 'mel'
	instruments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

	def __init__(self, mode='mel'):
		"""
		Parameters:
			mode: The type of features to create from the dataset. Available values:
				'mel' for mel frequency
				'handpicked' for the variable features mentionedi in the documention
		"""
		allowed_values = ['mel', 'handpicked']
		if (mode not in allowed_values):
			print('Mode can only be "mel" or "handpicked". Falling back to default "mel"')
			self.mode = 'mel'
		else:
			self.mode = mode

	def normalizeGain(self, input_path, dB=-15):
		""" Normalizes the gain of the dataset to a specific dB level
		THIS WILL REPLACE YOUR DATASET. Remember to make a copy of the folder if you want to keep the original.
		Parameters:
			input_path: The folder of the train or test set
			dB: The dB value to normalize the tracks to. Default -15
		"""
		print("Normalizing gain of {} to {}dB".format(input_path, dB))
		total = 0
		if os.path.isdir(input_path):
			total = self._normalizeGainFolder(input_path, dB)
		elif self._normalizeGainFile(input_path, dB):
			total = 1
		print("Normalized {} .wav files\t\t".format(total))

	def _normalizeGainFolder(self, input_dir, dB):
		""" Normalizes all tracks in the given folder to the specified dB gain. Traverses to subfolders
		THIS WILL REPLACE THE CONTENTS OF THE GIVEN FOLDER.
		"""
		folder_contents = os.listdir(input_dir)
		total_normalized = 0

		for filename in folder_contents:
			full_filename = os.path.join(input_dir, filename)
			if os.path.isdir(full_filename):
				total_normalized += self._normalizeGainFolder(full_filename, dB)
			elif self._normalizeGainFile(full_filename, dB):
				total_normalized += 1
		return total_normalized

	def _normalizeGainFile(self, filename, dB):
		""" Normalizes the given track to the specified dB gain. Traverses to subfolders
		THIS WILL REPLACE THE GIVEN TRACK.
		"""
		if not filename.endswith('.wav'):
			return False
		print('Normalizing {}'.format(os.path.basename(filename)[:15]), end="\r")
		sys.stdout.flush()

		sound = AudioSegment.from_file(filename, 'wav')
		change_in_dBFS = dB - sound.dBFS
		sound = sound.apply_gain(change_in_dBFS)
		sound.export(filename, format="wav")
		return True

	def generateTrain(self, data_path, normalize_features=True):
		""" Creates a .h5 file containing the trainset. The resulting filename follows the pattern train_<mode>_<normalized>.h5
		Parameters:
			data_path: The folder that containts the trainset. It should contain a folder for each instrument
			normalize_features: Apply MinMax normalization to the features
		"""

		output_path = 'train_{}'.format(self.mode)
		if normalize_features:
			output_path += '_normalized'
		output_path += '.h5'

		print("Creating h5 from {} to file {}".format(data_path, output_path))

		if self.mode == 'mel':
			feature_x_size = 128
		elif self.mode == 'handpicked':
			feature_x_size = 25
		feature_y_size = 130

		labels = os.listdir(data_path)
		labels = list(set(self.instruments).intersection(labels))
		labels.sort()
		total_tracks = self._countTrainTracks(data_path, labels)
		data_matrix = np.empty((total_tracks, feature_x_size, feature_y_size))
		data_labels = np.chararray((total_tracks, 1), itemsize=4, unicode=True)
		index = 0

		for l, label in enumerate(labels):
			print("Data for {}".format(label))
			instrument_dir = os.path.join(data_path, label)
			files = os.listdir(instrument_dir)
			skipped = 0

			for i, track in enumerate(files):
				print(" {} of {}".format(i+1,len(files)), end="\r")
				sys.stdout.flush()

				try:
					y, sr = librosa.load(os.path.join(instrument_dir, track))

					if self.mode == 'mel':
						data_matrix[index] = librosa.amplitude_to_db(librosa.feature.melspectrogram(y, sr))
					elif self.mode == 'handpicked':
						data_matrix[index, 0] = librosa.feature.spectral_centroid(y, sr)
						data_matrix[index, 1] = librosa.feature.spectral_bandwidth(y, sr)
						data_matrix[index, 2] = librosa.feature.spectral_rolloff(y, sr)
						data_matrix[index, 3] = librosa.feature.zero_crossing_rate(y)
						data_matrix[index, 4] = librosa.feature.rmse(y)[0]
						data_matrix[index, 5:25] = librosa.feature.mfcc(y, sr, n_mfcc=20)
					data_labels[index] = label
					index += 1
				except (KeyboardInterrupt, SystemExit):
					raise
				except:
					print('err')
					skipped += 1
			print("")

			if skipped > 0:
				print(" Skipped {} corrupted files".format(skipped))
				data_matrix = data_matrix[:total_tracks-skipped]
				data_labels = data_labels[:total_tracks-skipped]

		# Open dataset file
		out_file = h5py.File(output_path, 'w')
		metadata_datasets = []

		# Normalise
		if normalize_features:
			minis = np.min(data_matrix, axis=0)
			maxis = np.max(data_matrix, axis=0)
			diff = maxis - minis
			data_matrix = np.nan_to_num(2 * ((data_matrix - minis) / diff) - 1)
			out_file.create_dataset('min', data=minis)
			metadata_datasets.append('min')
			out_file.create_dataset('max', data=maxis)
			metadata_datasets.append('max')

		# Write to file
		for l, label in enumerate(labels):
			out_file.create_dataset(label, data=data_matrix[(data_labels == label).flatten()])
		out_file.attrs['vector_size'] = (feature_x_size, feature_y_size)
		out_file.attrs['metadata_datasets'] = metadata_datasets
		out_file.close()

		print("Done")

	def generateTest(self, data_path, normalize_features=True):
		""" Creates a .h5 file containing the testset. The resulting filename follows the pattern test_<mode>_<normalized>.h5
		Call generateTrain before calling generateTest
		Parameters:
			data_path: The folder that containts the testset. It should contain all .wav and .txt files
			normalize_features: Apply MinMax normalization to the features using the MinMax values found in the trainset
		"""

		output_path = 'test_{}'.format(self.mode)
		if normalize_features:
			output_path += '_normalized'
		output_path += '.h5'

		if not os.path.isfile(output_path.replace('test', 'train')):
			raise ValueError('There is not trainset to get the MinMax values. Please call generateTrain first or set normalize_features=False');

		print("Creating h5 from {} to file {}".format(data_path, output_path))

		if self.mode == 'mel':
			feature_x_size = 128
		elif self.mode == 'handpicked':
			feature_x_size = 25
		feature_y_size = 130
		num_of_tracks = 3 * int(len(os.listdir(data_path)) / 2)

		data_matrix = np.empty((num_of_tracks, feature_x_size, feature_y_size), dtype=np.float32)
		print(data_matrix.shape)
		data_labels = np.empty((num_of_tracks, len(self.instruments)), dtype=bool)
		data_titles = np.chararray((num_of_tracks,1), itemsize=80)
		index = 0
		skipped = 0

		for i, track in enumerate(os.listdir(data_path)):
			if not track.endswith('wav'):
				continue
			print("Track {}".format(index), end="\r")
			sys.stdout.flush()
			track_path = os.path.join(data_path, track)

			try:
				y, sr = librosa.load(track_path)
				if self.mode == 'mel':
					track_data = librosa.amplitude_to_db(librosa.feature.melspectrogram(y, sr))
				elif self.mode == 'handpicked':
					f1 = librosa.feature.spectral_centroid(y, sr)[0]
					f2 = librosa.feature.spectral_bandwidth(y, sr)[0]
					f3 = librosa.feature.spectral_rolloff(y, sr)[0]
					f4 = librosa.feature.zero_crossing_rate(y)[0]
					f5 = librosa.feature.rmse(y)[0]
					f6 = librosa.feature.mfcc(y, sr, n_mfcc=20)

				for j in range(3):
					if self.mode == 'mel':
						if track_data[:,j*130:(j+1)*130].shape[1] < 130:
							break
						data_matrix[index] = track_data[:,j*130:(j+1)*130]
						data_titles[index] = track.replace('.wav', '')
						data_labels[index] = self._getTestOneHotLabels(os.path.join(data_path, track.replace('.wav', '') + '.txt'))
						index += 1
					elif self.mode == 'handpicked':
						if (f1[j*130:(j+1)*130].shape[0] < 130):
							break
						data_matrix[index, 0] = f1[j*130:(j+1)*130]
						data_matrix[index, 1] = f2[j*130:(j+1)*130]
						data_matrix[index, 2] = f3[j*130:(j+1)*130]
						data_matrix[index, 3] = f4[j*130:(j+1)*130]
						data_matrix[index, 4] = np.pad(f5[j*130:j*130 + 126], 2, 'constant')
						data_matrix[index, 5:25] = f6[:, j*130:(j+1)*130]
						data_titles[index] = track.replace('.wav', '')
						data_labels[index] = self._getTestOneHotLabels(os.path.join(data_path, track.replace('.wav', '') + '.txt'))
						index += 1

			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				skipped += 1
		print("")

		if skipped > 0:
			print(" Skipped {} corrupted files".format(skipped))
			data_matrix = data_matrix[:index]
			data_labels = data_labels[:index]
			data_titles = data_titles[:index]

		# Normalise
		out_file = h5py.File(output_path, 'w')
		metadata_datasets = []

		# Normalise
		if normalize_features:
			in_file = h5py.File(output_path.replace('test', 'train'), 'r')
			maxis = np.array(in_file['max'])
			minis = np.array(in_file['min'])
			in_file.close()
			diff = maxis - minis
			data_matrix = np.nan_to_num(2 * ((data_matrix - minis) / diff) - 1)

		# Write to file
		out_file.create_dataset('features', data=data_matrix)
		out_file.create_dataset('labels', data=data_labels)
		out_file.create_dataset('track_titles', data=data_titles)
		metadata_datasets.append('track_titles')
		out_file.attrs['instruments'] = self.instruments
		out_file.attrs['vector_size'] = (feature_x_size, feature_y_size)
		out_file.attrs['metadata_datasets'] = metadata_datasets
		out_file.close()

		print("Done")

	def _getTestOneHotLabels(self, track_path):
		""" Creates an onehot array from a .txt file in the testset
		"""
		with open(track_path.replace('\0',''),'r') as f:
			content = f.readlines()
		labels = [x.strip() for x in content]

		labels_arr = np.zeros(len(self.instruments), dtype=bool)
		for l in labels:
			labels_arr[self.instruments.index(l)] = True
		return labels_arr

	def _countTrainTracks(self, input_path, labels):
		""" Counts the number of tracks in the folders of the trainset
		"""
		total = 0
		for l, label in enumerate(labels):
			instrument_dir = os.path.join(input_path, label)
			total += len(os.listdir(instrument_dir))

		return total

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print("Usage: python DatasetPreprocessor.py train_path test_path mode")
		exit()
	dp = DatasetPreprocessor(sys.argv[3])
	dp.generateTrain(sys.argv[1])
	dp.generateTest(sys.argv[2])
