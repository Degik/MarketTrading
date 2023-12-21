import os
import torch
import NetMarket
import LoadData
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import IPython.display as display
from itertools import product

print("TRAINING MARKET V1.0")
# HYPERPARAMETER
#interval = 0.25
#learning_rate = 0.8
#hidden_units = 4
num_epochs = 500
momentum = 0.8
#GRID SEARCH PARAMETERS
intervals = [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]
learning_rates = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]


# PATH
pathTrain = "datasets/dataTrain.csv"
pathTest = "datasets/dataTest.csv"

# IMPORT DATA
data = LoadData.Data(pathTrain, pathTest)
# DATA: TENSOR, GPU, DATALOADER
data.convertToTensor()
data.moveToGpu()
data_loader_train, data_loader_test = data.createDataLoader()

for i, (interval, learning_rate) in enumerate(product(intervals, learning_rates)):
    
    # CREATE NET
    # Regressor Net
    net = NetMarket.NetMarket(interval)
    # MOVE NET TO GPU
    net = net.to("cuda:0")
    # SET TYPE NET
    net = net.double()

    # OPTIMIZER AND CRITERION
    print("Load MSELoss [criterion]\nLoad SGD [optimizer]")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    
    pathName = f'models/test_{i+1}-{interval}-{learning_rate}'
    # CREATE DIR
    os.makedirs(pathName, exist_ok=True)
    # MODEL SAVE
    torch.save(net, f'{pathName}/model.pth')
    with open(f'{pathName}/model_parameters.txt', 'w') as file:
        file.write('Pesi layer1\n')
        file.write(str(net.layer1.weight.data) + '\n')
        file.write('Bias layer1\n')
        file.write(str(net.layer1.bias.data) + '\n')
        file.write('Pesi layer2\n')
        file.write(str(net.layer2.weight.data) + '\n')
        file.write('Bias layer2\n')
        file.write(str(net.layer2.bias.data) + '\n')
        file.write('Pesi layer3\n')
        file.write(str(net.layer3.weight.data) + '\n')
        file.write('Bias layer3\n')
        file.write(str(net.layer3.bias.data) + '\n')


    #Values used for graphs
    loss_values_train = []
    accuracy_values_train = []
    loss_values_test = []
    accuracy_values_test = []
    # BEST
    best_accuracy_train = 0.0
    best_accuracy_test = 0.0

    net.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total = 0
        correct = 0
        
        for batch_input, batch_output in data_loader_train:
            #Forward pass
            outputs = net(batch_input)
            print(outputs)
            #Training loss
            loss = criterion(outputs, batch_output)
            #Calculate total loss
            total_loss += loss.item()
            #Backward and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Coefficient of Determination (R²)
            ss_tot = torch.sum((batch_output - torch.mean(batch_output)) ** 2)
            ss_res = torch.sum((batch_output - outputs) ** 2)
            r2 = 1 - ss_res / ss_tot
        best_accuracy_train = max(best_accuracy_train, r2)
        avg_loss = total_loss / len(data_loader_train)
        #Add to list
        loss_values_train.append(avg_loss)
        accuracy_values_train.append(r2)


        total = 0
        correct = 0
        #CALCULATE ACCURACY VAL
        net.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_input, batch_output in data_loader_test:
                outputs = net(batch_input)
                loss = criterion(outputs, batch_output)
                total_loss += loss.item()
                #predicted = torch.sign(outputs)
                # Coefficient of Determination (R²)
                ss_tot = torch.sum((batch_output - torch.mean(batch_output)) ** 2)
                ss_res = torch.sum((batch_output - outputs) ** 2)
                r2 = 1 - ss_res / ss_tot
            best_accuracy_test = max(best_accuracy_test, r2)
            avg_loss = total_loss / len(data_loader_test)
            #Add to list
            loss_values_test.append(avg_loss)
            accuracy_values_test.append(r2)
        net.train()

    #Save plot loss
    display.clear_output(wait=True)
    plt.plot(loss_values_train, label='Training Loss')
    plt.plot(loss_values_test, label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss for Epoch')
    plt.legend()
    plt.savefig(f'{pathName}/Loss.png')
    plt.clf()

    #Save plot accuracy
    display.clear_output(wait=True)
    plt.plot(accuracy_values_train, label='Accuracy Train')
    plt.plot(accuracy_values_test, label='Accuracy Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Coefficient of Determination (R²) for Epoch')
    plt.legend()
    plt.savefig(f'{pathName}/Accuracy-test.png')
    plt.clf()

    result = f'Test: {i+1} - Interval:{interval}, Learning-rate: {learning_rate}, Best-R²-Train: {best_accuracy_train:.4f}, Best-R²-Test: {best_accuracy_test:.4f}'
    with open(f"resultTest.txt", 'w') as file:
        file.write(result + "\n")
    print(result)