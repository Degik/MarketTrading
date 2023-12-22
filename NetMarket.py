import torch
import torch.nn as nn
import torch.nn.functional as F

class NetMarket(nn.Module):
    def __init__(self, interval:float):
        super(NetMarket, self).__init__()
        
        #Layer 1 Input: 3 Output: 9
        self.layer1 = nn.Linear(3, 5)
        nn.init.uniform_(self.layer1.weight, -interval, interval)
        #nn.init.constant_(self.layer1.bias, 0.2)
        nn.init.zeros_(self.layer1.bias)
        #Layer 2 Input: 9 Output: 27
        self.layer2 = nn.Linear(5, 1)
        nn.init.uniform_(self.layer2.weight, -interval, interval)
        #nn.init.constant_(self.layer2.bias, 0.2)
        nn.init.zeros_(self.layer2.bias)


    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(3, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 0)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class LSTMNet(nn.Module):
    def __init__(self, input_size=4, hidden_size1=64, hidden_size2=32, output_size=3):
        super(LSTMNet, self).__init__()
        
        self.bi_lstm1 = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size1, 
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(0.2)
        
        self.bi_lstm2 = nn.LSTM(
            input_size=hidden_size1*2,  # Because it's bidirectional
            hidden_size=hidden_size2,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size2*2, output_size)  # Output layer

    def forward(self, x):
        x, _ = self.bi_lstm1(x)
        x = self.dropout(x)
        x, _ = self.bi_lstm2(x)
        
        # Selecting the output of the last time step
        x = x[:, -1, :]
        
        x = self.fc(x)
        return x
