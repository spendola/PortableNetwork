import numpy as np
import os
import random
import urllib.request

class DataLoader(object):
	
	# Initializes the fully connected neural network
	def __init__(self):
		print("initializing data loader")
		self.localPath = os.path.dirname(os.path.realpath(__file__))
		self.dataPath = os.path.join(self.localPath, "data")
		self.precision = 5
		
	def LoadCsvFile(self, filepath, firstRow=0):
		print("Loading CSV file to Dataset")
		with open(os.path.join(self.dataPath, filepath)) as file:
			content = file.readlines()
		lines = [x.strip() for x in content]
		
		data = []
		for line in lines[firstRow:]:
			values = line.split(",")
			data.append(values)
		return data
		
	def ExpandCsvFile(self, filepath, ratio=10, noise=0.1, verbose=False):
		print("Expanding CSV file")
		expanded = []
		data = self.LoadCsvFile(filepath, firstRow=1)
		
		if(verbose):
			print("\nOriginal Data")
			for row in data:
				print(row)
				
		for row in data:
			expanded.append(row)
			for i in range(ratio):
				temp = [row[0], row[1], row[2]]
				temp = temp + [round(float(x) * float(random.uniform(1.0 - noise, 1.0 + noise)), self.precision) for x in row[3:-2]]
				temp.append(row[-2])
				temp.append(row[-1])
				expanded.append(temp)
		
			
		with open(os.path.join(self.dataPath, "expanded_dataset.csv"), "w+") as output:
			for row in expanded:
				output.write(", ".join([str(x) for x in row]) + "\n")
				print(row)
				
		return expanded
		
	def BuildDataset(self, start, end):
		dataset, weeks = [], []
		
		#read weeks and expected outputs
		with open(os.path.join(self.dataPath, "data_external.txt")) as file:
			content = file.readlines()
		lines = [x.strip() for x in content]
		for line in lines:
			weeks.append(line.replace(" ", "").split(","))
		
		# read data specifications
		with open(os.path.join(self.dataPath, "data_specification.txt")) as file:
			content = file.readlines()
		lines = [x.strip() for x in content]
		
		# download all values from database
		for week in weeks:
			temp = []
			temp.append(str(week[0]))
			temp.append(str(week[1]))
			total = self.UrlRequest("total", week[0], week[1], "total", True)
			if(int(total) == 0):
				print("skipping week " + str(week[0]) + ": insufficient data")
				continue
			temp.append(total) 
			for line in lines:
				data = round(int(self.UrlRequest("count", week[0], week[1], line, True)) / float(total), self.precision)
				temp.append(str(data))
			temp.append(str(week[2]))
			temp.append(str(week[3]))
			dataset.append(temp)
			print(temp)
		
		with open(os.path.join(self.dataPath, "new_dataset.csv"), "w+") as output:
			for data in dataset:
				output.write(", ".join(data) + "\n")
			
		return dataset
		
	def LoadDataset(self, path, evaluation_size=100):
		data = self.LoadCsvFile(path)
		samples = np.zeros((len(data), len(data[0]) - 2), dtype=float)
		labels = np.zeros((len(data), 2), dtype=float)

		for sample in data:
			row = data.index(sample)
			labels[row][0] = sample[-2]
			labels[row][1] = sample[-1]
			for i in range(len(sample)-2):
				samples[row][i] = sample[i]
			
		print("creating training dataset")
		print(samples.shape)
		data = [np.reshape(x, (5, 1)) for x in samples[:-evaluation_size]]
		label = [np.reshape(x, (2, 1)) for x in labels[:-evaluation_size]]
		trainingData = list(zip(data, label))
		
		print("creating evaluation dataset")
		data = [np.reshape(x, (5, 1)) for x in samples[-evaluation_size:]]
		label = [np.reshape(x, (2, 1)) for x in labels[-evaluation_size:]]
		evaluationData = list(zip(data, label))
		
		print("Training Data: " + str(len(trainingData)))
		print("Evaluation Data: " + str(len(evaluationData)))
		return trainingData, evaluationData
		
	def GenerateTestDataset(self):
		print("Generating Sample Dataset")
		with open(os.path.join(self.dataPath, "sample_dataset.csv"), "w+") as output:
			for i in range(1000):
				a = round(random.uniform(0.0, 100.0), 2)
				b = round(random.uniform(0.0, 100.0), 2)
				c = round(random.uniform(0.0, 100.0), 2)
				d = round(random.uniform(0.0, 100.0), 2)
				e = round(random.uniform(0.0, 100.0), 2)
				f = round(((a) + (b*2) + (c*3) + (d*4) + (e*5)) / 1500.0, 3)
				g = round(1.0 - f, 3)
				output.write(str(a) + ", " + str(b) + ", " + str(c) + ", " + str(d) + ", " + str(e) + ", " + str(f) + ", " + str(g) + "\n")
		print("done")
		
	def UrlRequest(self, type, week, year, resource, verbose=False):
		post_data = urllib.parse.urlencode({'type':type, 'week':str(week), 'year':str(year), 'args':resource}).encode('utf-8')		
		x = urllib.request.urlopen(url='http://www.pendola.net/api/datalink.php', data=post_data)
		data = x.read().decode("utf-8")
		if(verbose):
			print("week " + str(week) + ", " + resource + " -> " + data)
		return data
	