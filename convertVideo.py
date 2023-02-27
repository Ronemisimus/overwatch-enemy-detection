import numpy as np
import tqdm
import torch
import torchvision
from PIL import Image
import cv2
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import nms
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import SSDTransfoms as ssd_tfms

video_path = "/home/ronen/Videos/taking my goat on a hike and hoping she doesn't headbutt children.mp4"
video_path_out = "/home/ronen/Videos/Goat-detection-res.mp4"
model_path = "models/classes4_frozen_up_to_backbone.features.1_goat.pkl"

label_dict = ["none","chicken", "gaurd dog", "goat"]
color_dict = [color for color in zip(range(3,255,63),[0]*4,range(255,3,-63))]
confedense = 0.55
nms_threshold = 0.4
show = False
frames_with_enemy = 0

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_path_out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))


model, loss = torch.load(model_path)

processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(processing_device)
model.eval()

test_transform = ssd_tfms.Compose([
        ssd_tfms.PILToTensor(),
        ssd_tfms.ConvertImageDtype(torch.float),
        ssd_tfms.ScaleToStandard()
    ])
totensor = torchvision.transforms.PILToTensor()
toPIL = torchvision.transforms.ToPILImage()
with torch.no_grad():
    last_shown = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(range(frame_count),
                     postfix={"enemy_frames":frames_with_enemy})
    for i in pbar:
        ret, frame = cap.read()
        if ret == True:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            conv_frame, _ = test_transform(frame_pil, None)
            conv_frame = conv_frame.to(processing_device)    
            pred = model([conv_frame])

            boxes:torch.Tensor = pred[0]['boxes'].detach().to(processing_device)
            scores:torch.Tensor = pred[0]['scores'].detach().to(processing_device)
            labels:torch.Tensor = pred[0]['labels'].detach().to(processing_device)

            height, width, _ = frame_rgb.shape

            ratio_x = width/320
            ratio_y = height/320

            boxes = torch.tensor([[box[0]*ratio_x, box[1]*ratio_y, box[2]*ratio_x, box[3]*ratio_y] for box in boxes],device=processing_device)

            box_num = int((scores>confedense).sum())

            if box_num>0:
                frames_with_enemy+=1
                boxes = boxes[scores>confedense]
                labels = labels[scores>confedense]
                scores = scores[scores>confedense]
                indecis = nms(boxes,scores,nms_threshold)
                boxes = boxes[indecis]
                labels_int = labels[indecis]
                scores_final = scores[indecis]

                labels = [label_dict[int(label)]+" "+str(int(score*100))+"%" for label, score in zip(labels_int,scores_final)]
                colors = [color_dict[int(label)] for label in labels_int]

                frame_tensor = totensor(frame_pil)
                frame = draw_bounding_boxes(frame_tensor,boxes=boxes,labels=labels,colors=colors,width=5)

                frame_pil: Image = toPIL(frame)
                pbar.postfix = {"enemy_frames":frames_with_enemy}
                if show and i - last_shown>10:
                    frame_pil.show()
                    last_shown = i
            
            out.write(
                cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            )
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    if out.isOpened():
        out.release()

