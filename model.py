import torchvision
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import MobileNet_V3_Large_Weights
import torch

def get_model():
    num_classes = 4
    # Step 1.
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        num_classes=num_classes,
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT)
    pretrained = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)

    # Step 2, load the model state_dict and the default model's state_dict
    mstate_dict = model.state_dict()
    cstate_dict = pretrained.state_dict()

    # Step 3.
    diffrent_shape_keys = []
    for k in mstate_dict.keys():
        if mstate_dict[k].shape != cstate_dict[k].shape:
            print('removed: key {}, orishape: {}, training shape: {}'.format(k, cstate_dict[k].shape, mstate_dict[k].shape))
            cstate_dict.pop(k)
            diffrent_shape_keys.append(k)
        elif cstate_dict[k].shape == torch.Size([]):
            if k[:-19]+"running_mean" not in cstate_dict:
                print('removed: key {}, orishape: {}, training shape: {}'.format(k, cstate_dict[k].shape, mstate_dict[k].shape))
                cstate_dict.pop(k)
                diffrent_shape_keys.append(k)
            else:
                print('not removed: key {}, orishape: {}, training shape: {}'.format(k, cstate_dict[k].shape, mstate_dict[k].shape))    
        else:
            print('not removed: key {}, orishape: {}, training shape: {}'.format(k, cstate_dict[k].shape, mstate_dict[k].shape))
            
    # Step 4.
    model.load_state_dict(cstate_dict, strict=False)
    
# "extra" in name or 
# "head" in name or 
# (int(name_arr[2])>=1) or 
# (int(name_arr[2])==0 and int(name_arr[3])>=11) 


    for name,param in model.named_parameters():
        name_arr = name.split(".")
        if "head" in name or \
        "extra" in name or \
        "backbone.features.1"<name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            print(f"freezing layer {name}")

    return model