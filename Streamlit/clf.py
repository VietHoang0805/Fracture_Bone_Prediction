# clf.py
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet121
from PIL import Image

class HandleTransparency(object):
    def __call__(self, img):
        if img.mode == 'RGBA' or img.mode == 'LA':
            img = img.convert('L')
        elif img.mode == 'P':
            img = img.convert('RGBA')
            img = img.convert('L')
        else:
            img = img.convert('L')
        return img

# Transforms for grayscale image
transform = transforms.Compose([
    HandleTransparency(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_model(model_name):
    if model_name == "ViT (Vision Transformer)":
        model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        model.patch_embed.proj = nn.Conv2d(
            in_channels=1,
            out_channels=model.patch_embed.proj.out_channels,
            kernel_size=model.patch_embed.proj.kernel_size,
            stride=model.patch_embed.proj.stride,
            padding=model.patch_embed.proj.padding,
            bias=False
        )
        in_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
        path = "Model/VIT/model.pt"

    elif model_name == "ResNet18":
        model = timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=2)
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=(model.conv1.bias is not None)
        )
        in_features = model.get_classifier().in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
        path = "Model/ResNet18/model.pt"

    elif model_name == "MobileNetV2":
        model = timm.create_model('mobilenetv2_100.ra_in1k', pretrained=True, num_classes=2)
        model.conv_stem = nn.Conv2d(
            in_channels=1,
            out_channels=model.conv_stem.out_channels,
            kernel_size=model.conv_stem.kernel_size,
            stride=model.conv_stem.stride,
            padding=model.conv_stem.padding,
            bias=(model.conv_stem.bias is not None)
        )
        in_features = model.get_classifier().in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
        path = "Model/MobileNet/model.pt"

    elif model_name == "DenseNet121":
        model = densenet121(pretrained=True)
        model.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=model.features.conv0.out_channels,
            kernel_size=model.features.conv0.kernel_size,
            stride=model.features.conv0.stride,
            padding=model.features.conv0.padding,
            bias=(model.features.conv0.bias is not None)
        )
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
        path = "Model/DenseNet/model.pt"

    else:
        raise ValueError("Invalid model name")

    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def predict_with_model(image, model_name):
    model = load_model(model_name)
    image = transform(image).unsqueeze(0)  # [1, 1, 224, 224]
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()
