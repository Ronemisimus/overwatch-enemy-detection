import numpy as np
import tqdm
import torch
import torchvision
from PIL import Image
import cv2
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

video_path = "/home/ronen/Videos/2023-02-21 11-26-27.mkv"
video_path_out = "/home/ronen/Videos/2023-02-21 11-26-27_res.mkv"
model_path = "models/latest_3classes.pkl"

label_dict = ["none","enemy","enemy"]
confedense = 0.2
show = False

cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(video_path_out,cv2.VideoWriter_fourcc('M','J','P','G'),60, (1920,1080))


model, loss = torch.load(model_path)

processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(processing_device)
model.eval()

transform = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.transforms()
totensor = torchvision.transforms.ToTensor()
toPIL = torchvision.transforms.ToPILImage()
with torch.no_grad():
    last_shown = 0
    frame_count = 14400 # int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm.tqdm(range(frame_count)):
        ret, frame = cap.read()
        if ret == True:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            conv_frame = transform(frame_pil).to(processing_device)    
            pred = model([conv_frame])

            boxes:torch.Tensor = pred[0]['boxes'].detach()
            scores:torch.Tensor = pred[0]['scores'].detach()
            labels:torch.Tensor = pred[0]['labels'].detach()

            box_num = int(torch.sum(scores>confedense))

            if box_num>0:
                boxes = boxes[scores>confedense]
                labels = labels[scores>confedense]

                labels = [label_dict[int(label)] for label in labels]

                frame_tensor = totensor(frame_pil)
                frame_tensor= (frame_tensor*255).round().to(torch.uint8)
                frame = draw_bounding_boxes(frame_tensor,boxes=boxes,labels=labels)

                frame_pil: Image = toPIL(frame)
                if show and i - last_shown>60:
                    frame_pil.show()
                    last_shown = i
            
            out.write(np.array(frame_pil))
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    if out.isOpened():
        out.release()

