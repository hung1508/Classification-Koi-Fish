from lib import *
from hyperparameters import *
from model import MyModel
from utils import *
from dataloader import *


print(f"Choose {device} to train") 

model = MyModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9) 

print(model)

for t in range(epochs): 
    print(f"Epoch {t+1}\n--------------------------")
    trainv2(model, dataloader_dict, loss_fn, optimizer, epochs)
    #valid(test_loader, model, loss_fn)
    if t% 5 ==0:
        name = "model_" + str(t) + ".pth"
        torch.save(model.state_dict(), os.path.join(model_path,name))
        print("Saved PyTorch Model State to " + name)
print('Done!')         