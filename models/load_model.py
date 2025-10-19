import argparse
import timm
import torch
import torch.nn as nn


def load_model(model_name):
    if model_name == "vit_base_patch16_224":
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )
    
    elif model_name == "resnet-50":
        model = timm.create_model("resnet50", pretrained=False, 
                checkpoint_path='./pretrained_model/resnet50-11ad3fa6.pth', num_classes=1000)
        state_dict = model.state_dict()
        keys_to_remove = [k for k in state_dict if "classifier" in k]
        for k in keys_to_remove:
            del state_dict[k]
        model_new = timm.create_model("resnet50", pretrained=False, num_classes=0)
        model_new.load_state_dict(state_dict, strict=False)
        return model_new
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")