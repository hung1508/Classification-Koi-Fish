from lib import *
from hyperparameters import *

transformer = transforms.Compose([transforms.Resize((partial_inputSize,partial_inputSize)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                  ])

train_data = ImageFolder(root=train_dir,transform=transformer)
test_data = ImageFolder(root=train_dir,transform=transformer)
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=False) 
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True) 


dataloader_dict = {"train": train_loader, 
                   "test": test_loader}