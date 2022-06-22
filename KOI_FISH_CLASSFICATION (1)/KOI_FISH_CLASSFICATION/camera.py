import cv2
from model import MyModel
from dataloader import transformer
from hyperparameters import classes, device, model_path
from lib import *


class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):

        model = MyModel().to(device)
        model.load_state_dict(torch.load(os.path.join(model_path,"best_model.pth"), map_location=torch.device(device)))
        ret, frame = self.video.read()
        # Display the resulting frame
        img = transformer(Image.fromarray(frame)).unsqueeze(0)
        pred = model(img)
        label = classes[pred.argmax(1)]
        
        cv2.putText(frame, "CLASS: " + str(label), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            
        ret,jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()