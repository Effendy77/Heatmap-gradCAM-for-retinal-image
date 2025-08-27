from __future__ import annotations
import torch, timm

def load_model(backbone: str, ckpt_path: str, num_classes: int = 1) -> torch.nn.Module:
    # Default: ViT B/16; swap in your RETFound builder if needed
    if backbone in ('vit', 'retfound_vit'):
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    elif backbone == 'resnet':
        import torchvision.models as tvm
        model = tvm.resnet50(weights=None)
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f'Unknown backbone: {backbone}')
    if ckpt_path and ckpt_path != 'NONE':
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f'[WARN] Non-strict load: {e}')
    model.eval()
    return model

def resolve_target_layer(model, dotted: str):
    # Resolve nested attribute with list indices, e.g., 'blocks.-1.norm1'
    obj = model
    for token in dotted.split('.'):
        if token.lstrip('-').isdigit():
            obj = obj[int(token)]
        else:
            obj = getattr(obj, token)
    return obj
