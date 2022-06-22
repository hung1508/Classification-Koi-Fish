from lib import *

train_dir = os.path.join("data","KoiFish","train")
test_dir = os.path.join("data","KoiFish","test")
model_path = 'model/'
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 70
partial_inputSize = 224
input_channels = 3
input_size = partial_inputSize*partial_inputSize*input_channels
output_size = 18 
lr = 1e-3

root=pathlib.Path(train_dir)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])