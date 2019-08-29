import torch
import torch.nn as nn
from _cpools import TopPool,BottomPool,RightPool,LeftPool


class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.pool1 = pool1()
        self.pool2 = pool2()
        self.relu1 = nn.ReLU(inplace=True)
        
    def forward(self,x):
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        out = self.relu1(pool1 + pool2)
        return x
        

        
class tl_pool(pool):
    def __init__(self, dim):
        super(tl_pool, self).__init__(dim, TopPool, LeftPool)
        



class Pool_model(nn.Module):
    def __init__(self):
        super(Pool_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.tl = tl_pool(64)
        
    def forward(self,x):
        
        conv1 = self.conv1(x)
        tl = self.tl(conv1)
        out = self.bn1(tl)
        return out
    
    
    
def Init():
    
    inp = torch.randn(1,3,512,512)
    
    net = Pool_model()
    
    ouputs = net(inp)
    
    print(ouputs.shape)
    
    
if __name__ == "__main__":
    
    Init()
        
    