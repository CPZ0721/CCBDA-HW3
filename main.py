import os
from PIL import Image
import utils
import torch
import numpy as np
from Unet2 import UNet
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from pytorch_gan_metrics import get_fid


def data_prepare(img_path):
    """
        DATA PREPARATION - Mnist 
    """
    train_img = os.listdir(img_path)
    train_img.sort()
    dataset = []
    transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
    for path in train_img:
        path = os.path.join(img_path, path)
        image = Image.open(path)
        img = transform(image)
        dataset.append(img)

    return dataset

def main():
    
    # data prepare
    img_path = './mnist'
    
    print("Data Preparing...")
    trainset = data_prepare(img_path=img_path)
    
    batch_size = 64
    # dataloader
    print("Create Data Loader...")
    train_dataset = utils.TransDataset(trainset)
    train_data_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True,  num_workers=8)

    # model setup and optimizer config
    n_steps = 1000
    beta = torch.linspace(0.0001, 0.04, n_steps)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    learning_rate = 1e-5
    epochs = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
    model = UNet().to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=0)
    min_train_loss = float("inf")
    # train
    print("Begin Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for img in train_data_loader:
            img = img.to(device)
            step = torch.randint(0, n_steps, (img.shape[0],), dtype=torch.long).cuda() # Random 't's 
            noise_img, noise = utils.generate_noise(img, step, alpha_bar) # Get the noised images and the noise
            pred_noise = model(noise_img.float(), step) 

            loss = F.mse_loss(noise.float(), pred_noise) 

            total_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step(loss)
        
        avg_train_loss = total_train_loss / len(train_data_loader)
        
        print('Epoch:%3d' % epoch, '|Train Loss:%8.4f ' % avg_train_loss)
        if avg_train_loss < min_train_loss:
            min_train_loss = avg_train_loss
            print("-------------saving model--------------")
            # save the model (include architecture, parameters)
            torch.save(model, "model.pth")
    

    
    # Sampling
    print("Start to generate images...")
    save_path = 'images'
    # generate the dir to put generated images
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = torch.load("model.pth").to(device)
    
    ims = []
    num_images = 10000
    BS = 100
    images = torch.zeros((num_images, 3, 32, 32)).cuda()  # Start with random noise
    images = torch.clamp(images, min=-1, max=1)  # clamp [-1, 1]
    images = (images + 1) / 2
    tf = transforms.Resize(28)
    for batch in range(0, num_images-BS, BS):
      x = images[batch:batch+BS,:,:,:]
      for i in range(n_steps):
          if batch == 0:
              if i%142 == 0:
                for j in range(8):
                  result = tf(x)
                  ims.append(result[j,:,:,:].cpu())
          t = torch.tensor(n_steps-i-1, dtype=torch.long).cuda()
          with torch.no_grad():
              pred_noise = final_model(x.float(), t.unsqueeze(0))
              x = utils.reverse_noise(x, pred_noise, t.unsqueeze(0), beta, alpha, alpha_bar)
                    
      for img in range(x.shape[0]):
        x = tf(x)
        file_name = str(batch +img + 1).zfill(5) + ".png"
        path = os.path.join(save_path,file_name)              
        save_image(x[img,:,:,:],path)
        
      print('-------saving--%d-----'%batch)
    
    for i in range(len(ims)):
      ims[i] = tf(ims[i])
    # Diffusion Process
    save_image(ims, 'process_result.png')

    # Output FID
    print('Calculate the FID...')
    images = []
    for i in range(1,num_images+1):
        path = os.path.join(f'./images/{i:05d}.png')
        image = read_image(path) / 255.
        images.append(image)
    images = torch.stack(images, dim=0)
    FID = get_fid(images,'mnist.npz')
    print(f'{FID:.5f}')

if __name__ == "__main__":
    main()



