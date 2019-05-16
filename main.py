# PortableNetwork

import dataloader as dataloader
import fcnetwork as fcnetwork

def main():
	print("Portable Artificial Neural Network")
	print("version 0.1")

	while(True):
		print("\n1) Build Dataset")
		print("2) Expand Dataset")
		print("3) Construct Dataset")
		print("3) Train Neural Network")
		print("4) Unit Test")
		choice = int(input("Seleccione una Opcion > "))
		if(choice == 1):
			dataLoader = dataloader.DataLoader()
			dataLoader.BuildDataset(1, 10)
		if(choice == 2):
			dataLoader = dataloader.DataLoader()
			dataLoader.ExpandCsvFile("new_dataset.csv", 3, verbose=True)


if __name__ == "__main__": 
	main()