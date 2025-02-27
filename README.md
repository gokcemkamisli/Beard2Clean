# Beard2Clean: Automated Beard Removal via Image-to-Image Translation

**Overview:**  
Beard2Clean is an image-to-image translation project that automatically removes beards from facial images using a Pix2Pix GAN architecture. The project leverages a custom dataset generated with Stable Diffusion and InstructPix2Pix pipelines and employs a UNet-based generator alongside a PatchGAN discriminator. It demonstrates advanced training techniques—such as label smoothing and extra generator updates—to achieve high-quality results even under limited computational resources.

**Key Features:**
- **Custom Dataset Generation:**  
  Utilizes Stable Diffusion and InstructPix2Pix pipelines with consistent seeding to generate 100 paired images (bearded and clean-shaven) for controlled transformations.
- **Efficient Training Pipeline:**  
  Implements a tailored training loop in PyTorch that addresses hardware constraints by using small batch sizes (4), reduced resolution (256×256), and optimized update strategies.
- **Performance Evaluation:**  
  Evaluated with quantitative metrics like SSIM and PSNR, as well as qualitative visual inspections to ensure realistic beard removal.
- **Scalability & Future Enhancements:**  
  Plans include training on higher resolutions, incorporating perceptual loss (e.g., VGG-based or LPIPS), and leveraging advanced architectures like Pix2PixHD when resources allow.

**Technologies Used:**
- Python, PyTorch, NumPy
- Diffusers (Stable Diffusion & InstructPix2Pix pipelines)
- Matplotlib for visualization

**Setup Instructions:**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gokcemkamisli/Beard2Clean.git
   cd Beard2Clean
