import timm
import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
# Thêm hàm để dự đoán hình ảnh bất kỳ

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
def predict_image(image_path):
    transforms = v2.Compose([
    HandleTransparency(),  # Thêm lớp xử lý chuyển đổi màu sắc
    v2.Grayscale(1),
    v2.Resize((224, 224)),
    v2.ToTensor(),
    ])
    model = timm.create_model("deit3_small_patch16_224.fb_in22k_ft_in1k", pretrained=True, num_classes=2)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.blocks[11].parameters():
        param.requires_grad = True

    model.head.requires_grad = True

    outc = model.patch_embed.proj.out_channels
    kernel_size = model.patch_embed.proj.kernel_size
    stride = model.patch_embed.proj.stride
    # Chỉnh sửa lớp patch_embed.proj để phù hợp với kênh đầu vào là 1 (grayscale)
    model.patch_embed.proj = nn.Conv2d(in_channels=1,
                                out_channels=outc,
                                kernel_size=kernel_size,
                                stride=stride)
    model.load_state_dict(torch.load('model.pt',map_location=torch.device('cpu')))
    model.to('cpu')
    # Mở và chuyển đổi ảnh
    image = image_path
    image = transforms(image).unsqueeze(0)  # Thêm batch dimension

    # Chuyển ảnh sang device và dự đoán
    image = image.to('cpu')
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()