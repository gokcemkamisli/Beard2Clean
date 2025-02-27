import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.beard_dataset import BeardDataset, transform
from models.unet_generator import UNetGenerator

DATASET_PATH = "dataset"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(tensor):
    tensor = tensor.clone().detach()
    tensor = tensor * 0.5 + 0.5  
    tensor = tensor.permute(1, 2, 0).numpy()
    return tensor

def test_model(generator, dataloader, device='cpu'):
    generator.eval()
    inputs, outputs, targets = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            input_img = batch['input'].to(device)
            target_img = batch['target'].to(device)
            fake_img = generator(input_img)
            inputs.append(input_img.cpu())
            outputs.append(fake_img.cpu())
            targets.append(target_img.cpu())
            
    for i in range(5):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(denormalize(inputs[i][0]))
        axs[0].set_title("Input (Bearded)")
        axs[1].imshow(denormalize(outputs[i][0]))
        axs[1].set_title("Output (Transformed)")
        axs[2].imshow(denormalize(targets[i][0]))
        axs[2].set_title("Target (Clean)")
        for ax in axs:
            ax.axis('off')
        plt.show()

def main():
    dataset = BeardDataset(DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    generator = UNetGenerator().to(DEVICE)
    generator.load_state_dict(torch.load("models/generator.pth", map_location=DEVICE))
    test_model(generator, dataloader, device=DEVICE)

if __name__ == "__main__":
    main()
