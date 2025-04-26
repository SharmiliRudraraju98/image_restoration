import torch
import torchvision.transforms as transforms
from PIL import Image

# Assume models/ and pretrained models are properly organized

# Load Real-ESRGAN model
def load_real_esrgan():
    from models.Real_ESRGAN.realesrgan import RealESRGAN
    model = RealESRGAN()
    model.load_weights('models/pretrained/RealESRGAN_x4plus.pth')
    model.eval()
    return model

# Load BSRGAN model
def load_bsrgan():
    from models.BSRGAN.bsrgan import BSRGAN
    model = BSRGAN()
    model.load_weights('models/pretrained/BSRGAN.pth')
    model.eval()
    return model

# Load SwinIR model
def load_swinir():
    from models.SwinIR.swinir import SwinIR
    model = SwinIR()
    model.load_weights('models/pretrained/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
    model.eval()
    return model

# Enhance image using Real-ESRGAN
def enhance_image_real_esrgan(input_image, model):
    transform = transforms.ToTensor()
    img_tensor = transform(input_image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    output_img = transforms.ToPILImage()(output.squeeze(0))
    return output_img

# Enhance image using BSRGAN
def enhance_image_bsrgan(input_image, model):
    transform = transforms.ToTensor()
    img_tensor = transform(input_image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    output_img = transforms.ToPILImage()(output.squeeze(0))
    return output_img

# Enhance image using SwinIR
def enhance_image_swinir(input_image, model):
    transform = transforms.ToTensor()
    img_tensor = transform(input_image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    output_img = transforms.ToPILImage()(output.squeeze(0))
    return output_img
