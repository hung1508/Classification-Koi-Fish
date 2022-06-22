from lib import *
from hyperparameters import *

def trainv1(dataloader, model, loss_fn, optimizer): 
    size = len(dataloader.dataset)
    #Set training mode for model 
    model.train()
    
    for batch, (img,label) in enumerate(dataloader):
        img,label = img.to(device), label.to(device)
        optimizer.zero_grad()

        #Compute prediction error
        with torch.set_grad_enabled(True): 
            pred = model(img)
            loss = loss_fn(pred,label)
            #Backpropagation 
            loss.backward() 
            optimizer.step()
            loss, current = loss.item(), batch * len(img)
        print(f"loss: {loss: >7f} [{current:>5d}/{size:>5d}]")


def trainv2(model, dataloader_dict, loss_fn, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    print("Device: ", device)
    
    for epoch in range(num_epochs+1):
        
        #Transfer network to GPU/CPU 
        model.to(device)
        
        #Accelerate speed to GPU
        torch.backends.cudnn.benchmark = True
        
        print(f"Epoch {epoch}/{num_epochs}")
        
        for phase in ["train", "test"]:
            if phase == "train": 
                model.train() 
            else: 
                model.eval() 
            
            epoch_loss = 0.0 
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == "train"): 
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]): 
                
                #Transfer tensors of inputs (img) and labels to device GPU/CPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #Set gradient of optimizer to zero for every batch
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs) # output's dimension 4x2
                    loss = loss_fn(outputs, labels) 
                    _, predicts = torch.max(outputs, 1)
                    
                    if phase == "train": 
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item()*inputs.size(0)
                    
                    epoch_corrects += torch.sum(predicts==labels) #dtype = integer
                   
            print(f"Epoch_loss in {phase} phase {epoch_loss}")        
            print(f"Epoch_correct in {phase} phase {epoch_corrects}")
            epoch_loss = epoch_loss/ len(dataloader_dict[phase].dataset) # <<< The number of image in train_dataset
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            print(f" ^ {phase} Loss: {epoch_loss:.4f} - Acc: {epoch_accuracy:.4f}")
            
            
def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    #Set evalation mode for model 
    model.eval() 
    
    test_loss, correct = 0,0 
    
    with torch.no_grad() :
        for img, label in dataloader: 
            img,label = img.to(device), label.to(device)
            pred = model(img)
            test_loss += loss_fn(pred,label).item() 
            correct += (pred.argmax(1) == label).type(torch.float).sum().item() 
            
        test_loss /= num_batches 
        correct /= size 
        print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
        
def test(model, img,classes_list,transformer): 
    img = Image.open(img)
    img = transformer(img).unsqueeze(0).to(device)
    pred = model(img)
    return classes_list[pred.argmax(1)]
