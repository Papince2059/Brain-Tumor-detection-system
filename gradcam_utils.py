from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torch
import numpy as np
import cv2

# Image size constant
IMG_SIZE = 224

# Define transformation for the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(img_path , transform):
    pil_image = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0)
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return input_tensor, pil_image, np_image

def generate_gradcam(model, input_tensor, np_image, target_layer, device):
    model.eval()
    input_tensor = input_tensor.to(device)
    np_image_resized = cv2.resize(np_image, (IMG_SIZE, IMG_SIZE))

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]

    visualization = show_cam_on_image(np_image_resized, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = outputs.argmax(dim=1).item()

    return visualization, pred_class

def overlay_heatmap(cam, original_img_path, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))
