import torchvision
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
import torch

def get_model():
    num_classes = 3
    # Step 1.
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        num_classes=num_classes,
        trainable_backbone_layers=2)
    pretrained = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        trainable_backbone_layers=2)

    # Step 2, load the model state_dict and the default model's state_dict
    mstate_dict = model.state_dict()
    cstate_dict = pretrained.state_dict()

    # Step 3.
    for k in mstate_dict.keys():
        if mstate_dict[k].shape != cstate_dict[k].shape:
            print('key {} will be removed, orishape: {}, training shape: {}'.format(k, cstate_dict[k].shape, mstate_dict[k].shape))
            cstate_dict.pop(k)
        else:
            if 'backbone.features' in k:
                cstate_dict[k].requires_grad = False
            elif cstate_dict[k].dtype == torch.float32:
                cstate_dict[k].requires_grad = True

    # Step 4.
    model.load_state_dict(cstate_dict, strict=False)
    return model