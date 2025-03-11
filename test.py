from model import *
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange
from PIL import Image
import torch
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import matplotlib.pyplot as plt
model = None
device = None


class TestDataset(Dataset):
    def __init__(self, no_cloud_path, with_cloud_path):
        self.no_cloud_path = no_cloud_path
        self.with_cloud_path = with_cloud_path
        self.no_cloud_images = sorted(os.listdir(self.no_cloud_path), key=lambda x: int(x.split('.')[0]))
        self.with_cloud_images = sorted(os.listdir(self.with_cloud_path), key=lambda x: int(x.split('.')[0]))
        self.length = min(len(self.no_cloud_images), len(self.with_cloud_images))

    def __len__(self):
        return self.length  # 使用两个数据集中较小的长度

    def __getitem__(self, index):
        file_name1 = self.no_cloud_images[index]  # 获取文件名
        file_name2 = self.with_cloud_images[index]
        file_path1 = os.path.join(self.no_cloud_path, file_name1)
        file_path2 = os.path.join(self.with_cloud_path, file_name2)
        no_cloud_image = cv2.imread(file_path1).astype(np.float32) / 255.0
        with_cloud_image = cv2.imread(file_path2).astype(np.float32) / 255.0
        b, g, r = cv2.split(no_cloud_image)
        no_cloud_image = cv2.merge([r, g, b])
        b1, g1, r1 = cv2.split(with_cloud_image)
        with_cloud_image = cv2.merge([r1, g1, b1])
        no_cloud_image = no_cloud_image.transpose((2, 0, 1))  # 调整通道维度顺序 (H, W, C) -> (C, H, W)
        with_cloud_image = with_cloud_image.transpose((2, 0, 1))

        return torch.from_numpy(no_cloud_image), torch.from_numpy(with_cloud_image), file_name2  # 返回文件名


if __name__ == '__main__':
    model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 2,),
        attention_resolutions=[],
    )
    model.load_state_dict(torch.load('./checkpoints/best_checkpoint.pth', map_location=torch.device('cpu')))
    model = model.cuda()
    model.eval()

    ### MSE scope of T-CLOUD test set
    MSE_max = 0.200704  # 设置最大MSE值
    MSE_min = 0.001024  # 设置最小MSE值
    no_cloud_path = 'Path of cloud-free images in the test set'
    with_cloud_path = 'Path of cloud images in the test set'
    output_folder = "./RESULT"
    test_data = TestDataset(no_cloud_path, with_cloud_path)

    test_dataloader = DataLoader(test_data, batch_size=2, num_workers=0, shuffle=False)
    # 随机查看图片是否对应
    # num_samples_to_display = 3
    # samples = np.random.choice(len(test_data), num_samples_to_display, replace=False)
    # plt.figure(figsize=(15, 3))
    # for i, sample_idx in enumerate(samples):
    #     no_cloud_image, with_cloud_image = test_data[sample_idx]
    #     plt.subplot(2, num_samples_to_display, i + 1)
    #     plt.imshow(np.transpose(no_cloud_image, (1, 2, 0)))
    #     plt.title('No Cloud')
    #     plt.axis('off')
    #     plt.subplot(2, num_samples_to_display, num_samples_to_display + i + 1)
    #     plt.imshow(np.transpose(with_cloud_image, (1, 2, 0)))
    #     plt.title('With Cloud')
    #     plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()
    start_time = time.time()
    timesteps = 600 #400 or 600
    for no_cloud_image, with_cloud_image, file_names in test_dataloader:
        no_cloud_image = no_cloud_image.cuda()
        with_cloud_image = with_cloud_image.cuda()
        mse = F.mse_loss(with_cloud_image, no_cloud_image)
        t = (mse - MSE_min) // ((MSE_max - MSE_min) / (timesteps - 1)) + 1
        t = torch.tensor(t, dtype=torch.long, device=with_cloud_image.device).cuda()
        timesteps_emd = t.unsqueeze(0)
        estimate_noise = model(with_cloud_image, timesteps_emd)
        estimate_img = with_cloud_image - estimate_noise

        for j in range(estimate_img.shape[0]):
            img = rearrange(estimate_img[j, ...].cpu().squeeze(0), 'c h w -> h w c')
            img = torch.clamp(img, 0, 1)  # Clip the values to the range [0, 1]
            img = (img.detach().numpy() * 255).astype('uint8')  # Scale to [0, 255]
            img = Image.fromarray(img)  # Convert to PIL Image
            save_path = os.path.join(output_folder, file_names[j])  # 使用原始文件名
            img.save(save_path)  # Save the image to the specified folder

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Total inference time: {inference_time:.4f} seconds")

    # 计算每张图像的平均推理时间
    average_inference_time = inference_time / len(test_dataloader.dataset)
    print(f"Average inference time per image: {average_inference_time:.4f} seconds")


