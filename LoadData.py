import math
import utils
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class Data:
    def __init__(self, pathTrain:str, pathTest:str) -> None:
        pass   
        ## TRAIN IMPORT DATASET
        self.x_train = utils.importDatasetX(pathTrain)
        self.y_train = utils.importDatasetY(pathTrain)
        ## TEST IMPORT DATASET
        self.x_test = utils.importDatasetX(pathTest)
        self.y_test = utils.importDatasetY(pathTest)
        
        
    def convertToTensor(self):
        # CONVERT DATA FOR TRAINING
        self.x_train = self.x_train.to_numpy().astype(np.float32)
        self.y_train = self.y_train.to_numpy().astype(np.float32)
        # CONVERT DATA FOR TESTING
        self.x_test = self.x_test.to_numpy().astype(np.float32)
        self.y_test = self.y_test.to_numpy().astype(np.float32)
        ## CONVERT TO TENSOR TRAIN SET
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        # CONVERT TO TENSOR
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
    
    def moveToGpu(self):
        # MOVE TENSOR TRAIN TO GPU
        self.x_train = self.x_train.to("cuda:0")
        self.y_train = self.y_train.to("cuda:0")
        # MOVE TENSOR TO GPU
        self.x_test = self.x_test.to("cuda:0")
        self.y_test = self.y_test.to("cuda:0")
        
    def createDataLoader(self) -> (DataLoader, DataLoader):
        # CREATE DATALOADER TRAIN
        size = self.x_train.size(0)
        batchTrain =  math.ceil(size/2)
        print("Batch size for training: ", batchTrain)
        dataset_train = TensorDataset(self.x_train, self.y_train)
        data_loader_train = DataLoader(dataset_train, batch_size=batchTrain, shuffle=True)
        # CREATE DATALOADER TEST
        size = self.x_test.size(0)
        batchTest = math.ceil(size/2)
        print("Batch size for testing: ", batchTest)
        dataset_test = TensorDataset(self.x_test, self.y_test)
        data_loader_test = DataLoader(dataset_test, batch_size=batchTest, shuffle=True)
        return data_loader_train, data_loader_test
