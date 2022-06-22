import cv2 
from lib import *
from hyperparameters import classes, device, model_path
from dataloader import transformer
from model import MyModel


def realTimeTest():
    model = MyModel().to(device)
    model.load_state_dict(torch.load(os.path.join(model_path,"best_model.pth"), map_location=torch.device(device)))
    vid = cv2.VideoCapture(0)
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        # Display the resulting frame
        img = transformer(Image.fromarray(frame)).unsqueeze(0)
        pred = model(img)
        label = classes[pred.argmax(1)]
        
        cv2.putText(frame, "CLASS: " + str(label), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vid.release()
    cv2.destroyAllWindows()
    