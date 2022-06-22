from lib import *
from hyperparameters import *

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            #Input shape (64,3,224,224)
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            #Shape: (64,32,112,112)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            #Shape: (64,64,112,112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), 
            #Shape: (64,64,56,56) 
        ) 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*56*56,120),
            nn.ReLU(),
            nn.Linear(120,output_size), 
         )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    
    