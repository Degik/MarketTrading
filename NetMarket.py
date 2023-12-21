import torch.nn as nn
import torch.nn.functional as F

class NetMarket(nn.Module):
    def __init__(self, interval:float):
        super(NetMarket, self).__init__()
        
        #Layer 1 Input: 3 Output: 9
        self.layer1 = nn.Linear(3, 9)
        nn.init.uniform_(self.layer1.weight, -interval, interval)
        #nn.init.constant_(self.layer1.bias, 0.2)
        nn.init.zeros_(self.layer1.bias)
        #Layer 2 Input: 9 Output: 27
        self.layer2 = nn.Linear(9, 27)
        nn.init.uniform_(self.layer2.weight, -interval, interval)
        #nn.init.constant_(self.layer2.bias, 0.2)
        nn.init.zeros_(self.layer2.bias)
        #Layer 3 Input: 27 Output: 1
        self.layer3 = nn.Linear(27, 1)
        #
        nn.init.uniform_(self.layer3.weight, -interval, interval)
        #
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x  = self.layer3(x)
        return x