# Class FCNetwork
# Fully Connected Artificial Neural Network
import datetime
import random
import numpy as np
import os

class CrossEntropyCost(object):
	
	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@staticmethod
	def delta(z, a, y):
		return (a-y)
		
class FCNetwork(object):
	
	# Initializes the fully connected neural network
	def __init__(self, sizes, cost=CrossEntropyCost):
		print("initializing fully connected neural network")
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.cost = CrossEntropyCost
		self.localPath = os.path.dirname(os.path.realpath(__file__))
		
		# initialize random weights
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
		
		# set monitoring 
		self.monitorTrainingCost = True
		self.monitorTrainingAccuracy = False
		self.monitorEvaluationCost = True
		self.monitorEvaluationAccuracy = False
		
	def Train(self, trainingData, validationData, learnRate, lmbda, batchSize, epochs=10):
		while(True):
			self.StochasticGradientDescent(trainingData, epochs, batchSize, learnRate, lmbda, evaluationData=validationData)
			self.SampleEvaluation(validationData)
			print("\nTraining completed. What do you wish to do?")
			print("1) Save network state")
			print("2) Continue training")
			print("3) Terminate")
			choice = int(input("> "))
			if(choice == 1):
				now = datetime.datetime.now()
				self.Save(os.path.join(self.dataPath, "ann_"+now.strftime("%Y-%m-%d%H-%M")))
			if(choice == 2):
				learnRate = float(input("Learn Rate: "))
				lmbda = float(input("Regularization Rate: "))
				epochs = int(input("Epochs: "))
				continue
			if(choice == 3):
				break
		
	# Returns the output of the network for a given input a
	def FeedForward(self, a):
		for b, w in zip(self.biases, self.weights):
			 a = Sigmoid(np.dot(w, a)+b)
		return a
		
	# This is the function that performs the actual training of the network
	def StochasticGradientDescent(self, trainingData, epochs, miniBatchSize, learningRate, lmbda = 0.0, evaluationData=None):								  
		n_data = len(evaluationData)
		n = len(trainingData)
		
		for j in range(epochs):
			random.shuffle(trainingData)
			miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]
			for miniBatch in miniBatches:
				self.UpdateMiniBatch(miniBatch, learningRate, lmbda, len(trainingData))
			print ("Epoch {} training complete (lr:{})".format(j, learningRate))

			
			if self.monitorTrainingCost:
				cost = self.TotalCost(trainingData, lmbda)
				print("Cost on training data: {}".format(cost))

			if self.monitorTrainingAccuracy:
				accuracy = self.Accuracy(trainingData, convert=True)
				print("Accuracy on training data: {} / {}".format(accuracy, n))

			if self.monitorEvaluationCost:
				cost = self.EvaluationCost(evaluationData)
				print("Cost on evaluation data: " + str(cost))

			if self.monitorEvaluationAccuracy:
				accuracy = self.AccuracyCnt(evaluationData)
				percentage = ((float)(accuracy)/(float)(n_data))*100.0
				print("Accuracy on evaluation data: {}% of sample Size {}".format(percentage, n_data))
	
	def UpdateMiniBatch(self, miniBatch, learningRate, lmbda, n):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		batchSize = len(miniBatch)
		for x, y in miniBatch:
			delta_nabla_b, delta_nabla_w = self.BackPropagate(x, y)
			if(delta_nabla_b != None):
				nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
				nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
			else:
				batchSize = batchSize - 1

		# Regularized Leaning Rules 
		self.weights = [(1-learningRate*(lmbda/n))*w-(learningRate/batchSize)*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(learningRate/batchSize)*nb for b, nb in zip(self.biases, nabla_b)]
	
	def BackPropagate(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# feed forward
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			 z = np.dot(w, activation)+b
			 zs.append(z)
			 activation = Sigmoid(z)
			 activations.append(activation)

		# feed backward
		delta = (self.cost).delta(zs[-1], activations[-1], y)
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = SigmoidPrime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)
		
	def Accuracy(self, data, convert=False):
		return 0.0
		if convert:
			results = [(np.argmax(self.FeedForward(x)), np.argmax(y)) for (x, y) in data]
			total = sum(int(x == y) for (x, y) in results)
		else:
			results = [(np.argmax(self.FeedForward(x)), y) for (x, y) in data]
			total = sum(int(x == y) for (x, y) in results)
		self.Expected(results)
		self.HistResults(results)	
		self.Confusion(results)		
		return total
		
	def EvaluationCost(self, data):
		total = 0.0
		for x, y in data:
			prediction = self.FeedForward(x)
			expectation = y
			total = total + (prediction[0] - expectation[0]) + (prediction[1] - expectation[1])
		return total / float(len(data))
		
	def TotalCost(self, data, lmbda, convert=False):
		cost = 0.0
		for x, y in data:
			a = self.FeedForward(x)
			if convert:
				y = VectorizedResults(y)
			cost += self.cost.fn(a, y)/len(data)
		cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def SampleEvaluation(self, data):
		print("\nSample Evaluation")
		for x, y in data[:min(10, len(data))]:
			prediction = self.FeedForward(x)
			expectation = y
			print(str(expectation[0]) + ", " + str(expectation[1]) + " -> " + str(prediction[0]) + ", " + str(prediction[1]))
		
	def Save(self, filename):
		data = {"sizes": self.sizes,
				"weights": [w.tolist() for w in self.weights],
				"biases": [b.tolist() for b in self.biases],
				"cost": str(self.cost.__name__)}
		f = open(filename, "w")
		json.dump(data, f)
		f.close
		print("network state saved to: " + filename)

	def Load(self, path=""):
		# Choose a file to load
		if(path == ""):
			mtime = lambda f: os.stat(os.path.join("data", f)).st_mtime
			files = list(sorted(os.listdir("data"), key=mtime))
			print("\nChoose a file to load:")
			print('\n'.join('{}: {}'.format(*k) for k in enumerate(files)))
			fileindex = int(sys.stdin.readline().rstrip("\n"))
			filename = "data/" + files[fileindex]
		else:
			filename = path
		
		# Load the file
		print("...loading file " + filename)
		f = open(filename, "r")
		data = json.load(f)
		f.close()
		cost = getattr(sys.modules[__name__], data["cost"])
		self.weights = [np.array(w) for w in data["weights"]]
		self.biases = [np.array(b) for b in data["biases"]]
		
### Miscellaneous Functions
def VectorizedResults(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e
	
def Sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def SigmoidPrime(z):
	return Sigmoid(z)*(1-Sigmoid(z))
