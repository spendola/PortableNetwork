# PortableNetwork

import dataloader as dataloader
import fcnetwork as fcnetwork

def main():
	print("Portable Artificial Neural Network")
	print("version 0.1")

	while(True):
		print("\n1) Build Dataset")
		print("2) Expand Dataset")
		print("3) Generate Sample Dataset")
		print("4) Load Dataset")
		choice = int(input("Seleccione una Opcion > "))
		if(choice == 1):
			dataLoader = dataloader.DataLoader()
			dataLoader.BuildDataset(1, 10)
		if(choice == 2):
			dataLoader = dataloader.DataLoader()
			dataLoader.ExpandCsvFile("new_dataset.csv", 3, verbose=True)
		if(choice == 3):
			dataLoader = dataloader.DataLoader()
			dataLoader.GenerateTestDataset()
		if(choice == 4):
			datasetname = input("Dataset: ")
			epochs = int(input("Epochs: "))
			learningRate = float(input("Learning Rate: "))
			regularizationRate = float(input("Regularization Rate: "))
			batchSize = int(input("Batch Size: "))
			dataLoader = dataloader.DataLoader()
			network = fcnetwork.FCNetwork([5, 60, 2], cost=fcnetwork.CrossEntropyCost);
			trainingData, evaluationData = dataLoader.LoadDataset("sample_dataset.csv")
			network.Train(trainingData, evaluationData, learningRate, regularizationRate, batchSize, epochs)


if __name__ == "__main__": 
	main()