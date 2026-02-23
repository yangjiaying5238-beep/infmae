import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os, random, time
import models_infmae_skip4

# ============ 配置 ============
CHECKPOINT_PATH = "/root/autodl-tmp/output_new/checkpoint-49.pth"  # 新权重
IMG_FOLDER      = "/root/autodl-tmp/dataset/dataset/Inf30"
OUTPUT_FOLDER   = "reconstruction_results_new"   # 新文件夹，和baseline区分
MASK_RATIO      = 0.75
TEST_NUM        = 500
SEED            = 42   # 和baseline一样的seed，保证同一批图片
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_MEAN = [0.425, 0.425, 0.425]
IMG_STD  = [0.200, 0.200, 0.200]

# ============ 收集图片（同baseline）============
img_list = []
for root, dirs, files in os.walk(IMG_FOLDER):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img_list.append(os.path.join(root, f))

random.seed(SEED)
random.shuffle(img_list)
test_list = img_list[:TEST_NUM]
print(f"测试集: {len(test_list)} 张")

# ============ 加载新模型 ============
model = models_infmae_skip4.__dict__['infmae_vit_base_patch16_dec512d8b']()
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt.get('model', ckpt), strict=False)
model.to(DEVICE).eval()
print("新模型加载完成")

mean_t = torch.tensor(IMG_MEAN).view(3,1,1).to(DEVICE)
std_t  = torch.tensor(IMG_STD ).view(3,1,1).to(DEVICE)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============ 评估 ============
psnr_list, ssim_list = [], []
t0 = time.time()

for idx, img_path in enumerate(test_list):
    img_name = os.path.basename(img_path)
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        continue

    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=MASK_RATIO)

    recon    = model.unpatchify(pred)
    mask_3d  = mask.unsqueeze(-1).repeat(1, 1, 16*16*3)
    mask_img = model.unpatchify(mask_3d)
    full     = x * (1 - mask_img) + recon * mask_img

    def to_np(t):
        return np.clip((t * std_t + mean_t).permute(1,2,0).cpu().numpy(), 0, 1)

    orig_np = to_np(x.squeeze(0))
    full_np = to_np(full.squeeze(0))

    pv = psnr(orig_np, full_np, data_range=1.0)
    sv = ssim(orig_np, full_np, channel_axis=2, data_range=1.0)
    psnr_list.append(pv)
    ssim_list.append(sv)

    if (idx+1) % 50 == 0 or idx == 0:
        print(f"[{idx+1}/{len(test_list)}] PSNR:{pv:.2f} SSIM:{sv:.4f}")

# ============ 汇总对比 ============
print("\n" + "="*45)
print("改进后结果：")
print(f"  平均 PSNR : {np.mean(psnr_list):.4f} dB")
print(f"  平均 SSIM : {np.mean(ssim_list):.6f}")
print("="*45)
print("Baseline：")
print(f"  平均 PSNR : 32.4199 dB")
print(f"  平均 SSIM : 0.894708")
print("="*45)
diff_psnr = np.mean(psnr_list) - 32.4199
diff_ssim = np.mean(ssim_list) - 0.894708
print(f"PSNR 变化: {diff_psnr:+.4f} dB")
print(f"SSIM 变化: {diff_ssim:+.6f}")