import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.beard_dataset import BeardDataset, transform
from models.unet_generator import UNetGenerator
from models.patchgan_discriminator import PatchGANDiscriminator

# parameter configuration
DATASET_PATH = "dataset"
BATCH_SIZE = 4
NUM_EPOCHS = 60
LEARNING_RATE = 3e-4
BETA = (0.5, 0.999)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(generator, discriminator, dataloader, num_epochs=50, device='cpu'):
    generator.to(device)
    discriminator.to(device)
    
    # two of loss functions
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    
    # adam optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETA)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETA)
    
    #LR schedulers: halve the LR every 30 epochs
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            input_img = batch['input'].to(device)
            target_img = batch['target'].to(device)
            batch_size = input_img.size(0)
            
            # Label smoothing
            valid = torch.full((batch_size, 1, 31, 31), 0.9, device=device)
            fake = torch.zeros((batch_size, 1, 31, 31), device=device)
            
          
            # train generator (twice for balancing )


            for _ in range(2):
                optimizer_G.zero_grad()
                fake_img = generator(input_img)
                pred_fake = discriminator(fake_img, input_img)
                
                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_L1 = criterion_L1(fake_img, target_img)
                loss_G = loss_GAN + 15 * loss_L1
                loss_G.backward(retain_graph=True)
                optimizer_G.step()
            
            # train discriminator
                
            optimizer_D.zero_grad()
            pred_real = discriminator(target_img, input_img)
            loss_real = criterion_GAN(pred_real, valid)
            
            # i used the updated generator for a fresh fake image
            fake_img = generator(input_img).detach()
            pred_fake = discriminator(fake_img, input_img)
            loss_fake = criterion_GAN(pred_fake, fake)
            
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} "
                      f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")
        
        scheduler_G.step()
        scheduler_D.step()
    
    print("Training complete.")
    
    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), "models/generator.pth")
    torch.save(discriminator.state_dict(), "models/discriminator.pth")

def main():
    dataset = BeardDataset(DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()
    train_model(generator, discriminator, dataloader, num_epochs=NUM_EPOCHS, device=DEVICE)

if __name__ == "__main__":
    main()
