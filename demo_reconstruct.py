import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import models_infmae_skip4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载模型
model_name = 'infmae_vit_base_patch16_dec512d8b'
checkpoint_path = "InfMAE.pth"

model = models_infmae_skip4.__dict__[model_name]()
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.to(device)
model.eval()
print("Model loaded.")

# 归一化参数（与训练一致）
mean = torch.tensor([0.425, 0.425, 0.425]).view(3, 1, 1).to(device)
std  = torch.tensor([0.200, 0.200, 0.200]).view(3, 1, 1).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.425, 0.425, 0.425],
                         std=[0.200, 0.200, 0.200]),
])

img_list = ["/root/autodl-tmp/00001.png"]
output_folder = "reconstruction_results"
os.makedirs(output_folder, exist_ok=True)

# # 递归找jpg（兼容ImageFolder结构）
# img_list = []
# for root, dirs, files in os.walk(img_folder):
#     for f in files:
#         if f.endswith(".jpg") or f.endswith(".png"):
#             img_list.append(os.path.join(root, f))
# img_list = img_list[:10]

for img_path in img_list:
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loss, pred, mask = model(img_tensor, mask_ratio=0.75)

    print(f"pred shape: {pred.shape}, mask shape: {mask.shape}")  # 调试用

    # 重建图像
    recon = model.unpatchify(pred)  # [1, 3, 224, 224]

    # 生成mask图（像素级）
    patch_size = 16
    mask_patch = mask.unsqueeze(-1).repeat(1, 1, patch_size * patch_size * 3)
    mask_img = model.unpatchify(mask_patch)  # [1, 3, 224, 224]

    # 合成：保留原图未被mask部分 + 重建被mask部分
    img_full = img_tensor * (1 - mask_img) + recon * mask_img

    # 反归一化，用于可视化和计算指标
    def denorm(t):
        return torch.clamp(t * std + mean, 0, 1)

    orig_np   = denorm(img_tensor.squeeze(0)).permute(1,2,0).cpu().numpy()
    full_np   = denorm(img_full.squeeze(0)).permute(1,2,0).cpu().numpy()
    recon_np  = denorm(recon.squeeze(0)).permute(1,2,0).cpu().numpy()
    recon_np  = np.clip(recon_np, 0, 1)

    # 指标计算
    psnr_value = psnr(orig_np, full_np, data_range=1.0)
    ssim_value = ssim(orig_np, full_np, channel_axis=2, data_range=1.0)
    print(f"{img_name} -> PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, Loss: {loss.item():.4f}")

    # 可视化：原图 / masked图 / 重建图
    masked_np = denorm((img_tensor * (1 - mask_img)).squeeze(0)).permute(1,2,0).cpu().numpy()
    masked_np = np.clip(masked_np, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_np);   axes[0].set_title("Original");    axes[0].axis("off")
    axes[1].imshow(masked_np); axes[1].set_title("Masked Input"); axes[1].axis("off")
    axes[2].imshow(full_np);   axes[2].set_title(f"Reconstruction\nPSNR:{psnr_value:.2f} SSIM:{ssim_value:.4f}"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"compare_{img_name}"))
    plt.close(fig)

print("Done. Results saved in:", output_folder)