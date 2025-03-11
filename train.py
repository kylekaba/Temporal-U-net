
import os
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from model import UNetModel
from ptflops import get_model_complexity_info

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TrainDataset(Dataset):
    def __init__(self, no_cloud_path, with_cloud_path):
        self.no_cloud_path = no_cloud_path
        self.with_cloud_path = with_cloud_path
        self.no_cloud_images = sorted(os.listdir(self.no_cloud_path), key=lambda x: int(x.split('.')[0]))
        self.with_cloud_images = sorted(os.listdir(self.with_cloud_path), key=lambda x: int(x.split('.')[0]))
        self.length = min(len(self.no_cloud_images), len(self.with_cloud_images))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        file_path1 = os.path.join(self.no_cloud_path, self.no_cloud_images[index])
        file_path2 = os.path.join(self.with_cloud_path, self.with_cloud_images[index])
        no_cloud_image = cv2.imread(file_path1).astype(np.float32) / 255.0
        with_cloud_image = cv2.imread(file_path2).astype(np.float32) / 255.0
        b, g, r = cv2.split(no_cloud_image)
        no_cloud_image = cv2.merge([r, g, b])
        b1, g1, r1 = cv2.split(with_cloud_image)
        with_cloud_image = cv2.merge([r1, g1, b1])
        no_cloud_image = no_cloud_image.transpose((2, 0, 1))
        with_cloud_image = with_cloud_image.transpose((2, 0, 1))

        return torch.from_numpy(no_cloud_image), torch.from_numpy(with_cloud_image)


class TrainDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(TrainDataLoader, self).__init__(*args, **kwargs)

class WrappedModel(torch.nn.Module):
    def __init__(self, model, timesteps):
        super(WrappedModel, self).__init__()
        self.model = model
        self.timesteps = timesteps

    def forward(self, x):
        return self.model(x, self.timesteps)
if __name__ == '__main__':

    no_cloud_path = 'Path of cloud-free images in the train set'
    with_cloud_path = 'Path of cloud-free images in the train set'
    train_data = TrainDataset(no_cloud_path, with_cloud_path)
    train_dataloader = TrainDataLoader(train_data, batch_size=4, num_workers=2, shuffle=True)
    ##T-CLOUD
    MSE_max = 0.237360
    MSE_min = 0.000754
    timesteps = 600

    model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 2,),
        attention_resolutions=[],

    ).cuda()
    # model.load_state_dict(
    #     torch.load('C:/Users/TQW/Desktop/u-kan/checkpoints/best_checkpoint.pth', map_location=torch.device('cpu')))

    # 计算模型参数量和 FLOPs
    timesteps_tensor = torch.tensor([1], dtype=torch.long, device='cuda')  # 创建时间步
    wrapped_model = WrappedModel(model, timesteps_tensor)  # 包装模型
    input_size = (3, 256, 256)  # 假设输入是 256x256 的 RGB 图像

    # 使用 ptflops 计算 FLOPs 和参数量
    flops, params = get_model_complexity_info(wrapped_model, input_size, as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {flops}")
    print(f"Total Parameters: {params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练参数
    epoch_size = 300
    train_losses = []
    best_loss = float('inf')

    # 记录总训练时间
    total_training_start = time.time()

    for epoch in range(epoch_size):
        epoch_start_time = time.time()
        epoch_one_avg_loss = 0

        for no_cloud_image, with_cloud_image in train_dataloader:
            no_cloud_image = no_cloud_image.cuda()
            with_cloud_image = with_cloud_image.cuda()

            mse = F.mse_loss(with_cloud_image, no_cloud_image)
            t = (mse - MSE_min) // ((MSE_max - MSE_min) / (timesteps - 1)) + 1
            t = torch.tensor(t, dtype=torch.long, device=with_cloud_image.device).cuda()
            timesteps_emd = t.unsqueeze(0)

            estimate_noise = model(with_cloud_image, timesteps_emd)
            predict_img = with_cloud_image - estimate_noise
            loss = F.mse_loss(predict_img, no_cloud_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_one_avg_loss += loss.item()

        # 计算平均损失
        epoch_one_avg_loss /= len(train_dataloader)
        train_losses.append(epoch_one_avg_loss)

        epoch_end_time = time.time()
        print(f'Epoch {epoch + 1}/{epoch_size} avg loss: {epoch_one_avg_loss:.6f} '
              f'| Time: {epoch_end_time - epoch_start_time:.2f}s')

        # 保存模型
        torch.save(model.state_dict(), f'./checkpoints/checkpoint_epoch_{epoch + 1}.pth')

        if epoch_one_avg_loss < best_loss:
            torch.save(model.state_dict(), './checkpoints/best_checkpoint.pth')
            best_loss = epoch_one_avg_loss
            print(f'New best model saved with avg loss: {epoch_one_avg_loss:.6f}')

    # 打印总训练时间
    total_training_end = time.time()
    total_training_hours = (total_training_end - total_training_start) / 3600  # 将秒转换为小时
    print(f"Total training time: {total_training_hours:.2f} hours")

    # 绘制损失曲线
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.savefig('training_loss_curve.png')
    plt.show()


