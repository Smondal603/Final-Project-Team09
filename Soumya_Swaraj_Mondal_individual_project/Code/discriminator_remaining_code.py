class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1 + 4, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

        self.label_embedding_ATM = nn.Embedding(num_classes_ATM, img_size * img_size)
        self.label_embedding_CR = nn.Embedding(num_classes_CR, img_size * img_size)
        self.label_embedding_FR = nn.Embedding(num_classes_FR, img_size * img_size)
        self.label_embedding_PS = nn.Embedding(num_classes_PS, img_size * img_size)

    def forward(self, img, labels_ATM, labels_CR, labels_FR, labels_PS):
        label_ATM_flat = self.label_embedding_ATM(labels_ATM).view(labels_ATM.size(0), 1, img.size(2), img.size(3))
        label_CR_flat = self.label_embedding_CR(labels_CR).view(labels_CR.size(0), 1, img.size(2), img.size(3))
        label_FR_flat = self.label_embedding_FR(labels_FR).view(labels_FR.size(0), 1, img.size(2), img.size(3))
        label_PS_flat = self.label_embedding_PS(labels_PS).view(labels_PS.size(0), 1, img.size(2), img.size(3))

        d_in = torch.cat((img, label_ATM_flat, label_CR_flat, label_FR_flat, label_PS_flat), dim=1)
        validity = self.model(d_in)
        return validity
    

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss().to(device)


# Lists to store loss values for plotting
g_losses = []
d_losses = []

def generate_sample_images(epoch, generator, latent_dim, num_classes_ATM, num_classes_CR, num_classes_FR, num_classes_PS):
    r, c = 1, 5  # 1 row and 6 columns for visualization

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate random noise for 5 images
    noise = torch.randn(r * c, latent_dim).to(device)

    # Randomly sample labels F and V for 5 images
    sampled_labels_ATM = torch.randint(0, num_classes_ATM, (r * c,)).to(device)
    sampled_labels_CR = torch.randint(0, num_classes_CR, (r * c,)).to(device)
    sampled_labels_FR = torch.randint(0, num_classes_FR, (r * c,)).to(device)
    sampled_labels_PS = torch.randint(0, num_classes_PS, (r * c,)).to(device)

    # Generate images using the generator
    with torch.no_grad():  # Disable gradients for faster generation, to enable gradients, comment this line
        gen_imgs = generator(noise, sampled_labels_ATM, sampled_labels_CR, sampled_labels_FR, sampled_labels_PS)
        
    # Rescale images from [-1, 1] to [0, 1] for visualization
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set up the figure and axes
    fig, axs = plt.subplots(r, c, figsize=(15, 3))  # Wider figure for 5 images in one row

    # Plot each image in the row
    for i in range(c):
        axs[i].imshow(gen_imgs[i, 0].cpu().detach().numpy(), cmap='gray')
        axs[i].set_title(
            f"ATM: {sampled_labels_ATM[i].item()}, CR: {sampled_labels_CR[i].item()}, "
            f"FR: {sampled_labels_FR[i].item()}, PS: {sampled_labels_PS[i].item()}",
            fontsize=7
        )
        axs[i].axis('off')
        
    # Display the figure
    plt.tight_layout()
    plt.show()

def plot_loss_curves(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Training loop
for epoch in range(epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    num_batches = 0
    
    for i, (real_imgs, ATM_values, CR_values, FR_values, PS_values) in enumerate(batches):
        # Train discriminator
        noise = torch.randn(real_imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(noise, ATM_values, CR_values, FR_values, PS_values)
        
        valid = torch.ones(real_imgs.size(0), 1).to(device)
        fake = torch.zeros(real_imgs.size(0), 1).to(device)

        # Discriminator loss
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, ATM_values, CR_values, FR_values, PS_values), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), ATM_values, CR_values, FR_values, PS_values), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # Generator loss
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(gen_imgs, ATM_values, CR_values, FR_values, PS_values), valid)
        g_loss.backward()
        optimizer_G.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        num_batches += 1

    # Store average losses for the epoch
    g_losses.append(epoch_g_loss / num_batches)
    d_losses.append(epoch_d_loss / num_batches)

    # Print progress and show samples
    if (epoch + 1) % 5 == 0:
        print(f"[Epoch {epoch + 1}/{epochs}] [D loss: {d_losses[-1]:.4f}] [G loss: {g_losses[-1]:.4f}]")
        generate_sample_images(epoch + 1, generator, latent_dim, num_classes_ATM, num_classes_CR, num_classes_FR, num_classes_PS)
        plot_loss_curves(g_losses, d_losses)

# Final loss curves
plot_loss_curves(g_losses, d_losses)



# Function to generate labels for ATM, CR, FR and V
def generate_labels(batch_size, num_classes_ATM, num_classes_CR, num_classes_FR, num_classes_PS, device):
    labels_ATM = torch.randint(0, num_classes_ATM, (batch_size,), device=device)
    labels_CR = torch.randint(0, num_classes_CR, (batch_size,), device=device)
    labels_FR = torch.randint(0, num_classes_FR, (batch_size,), device=device)
    labels_PS = torch.randint(0, num_classes_PS, (batch_size,), device=device)
    return labels_ATM, labels_CR, labels_FR, labels_PS
    #if above line shows error, you can use the below
    #return torch.LongTensor(labels_ATM).to(device), torch.LongTensor(labels_CR).to(device), torch.LongTensor(labels_FR).to(device), torch.LongTensor(labels_PS).to(device)

# Save the generator and discriminator
torch.save(generator.state_dict(), 'generator_256.pth')
torch.save(discriminator.state_dict(), 'discriminator_256.pth')
# Optionally, save the optimizer states if you want to continue training
torch.save(optimizer_G.state_dict(), 'optimizer_G_256.pth')
torch.save(optimizer_D.state_dict(), 'optimizer_D_256.pth')

# Initialize the models (they should be created with the same architecture)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Load the saved state dictionaries
generator.load_state_dict(torch.load('generator.pth'))
discriminator.load_state_dict(torch.load('discriminator.pth'))

# If you want to continue training, also load the optimizers
optimizer_G.load_state_dict(torch.load('optimizer_G.pth'))
optimizer_D.load_state_dict(torch.load('optimizer_D.pth'))

# Initialize the models (they should be created with the same architecture)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Load the saved state dictionaries
generator.load_state_dict(torch.load('generator.pth'))
discriminator.load_state_dict(torch.load('discriminator.pth'))

# If you want to continue training, also load the optimizers
optimizer_G.load_state_dict(torch.load('optimizer_G.pth'))
optimizer_D.load_state_dict(torch.load('optimizer_D.pth'))

def generate_specific_image(ATM_value, CR_value, FR_value, PS_value, generator, latent_dim, device):
   
    # Generate random noise
    noise = torch.randn(1, latent_dim, device=device)
# Create labels as tensors
    ATM_label = torch.tensor([ATM_value], dtype=torch.long, device=device)
    CR_label = torch.tensor([CR_value], dtype=torch.long, device=device)
    FR_label = torch.tensor([FR_value], dtype=torch.long, device=device)
    PS_label = torch.tensor([PS_value], dtype=torch.long, device=device)

    # Generate the image using the generator
    with torch.no_grad():
        gen_img = generator(noise, ATM_label, CR_label, FR_label, PS_label).cpu()
    
    # Rescale image from [-1, 1] to [0, 1]
    gen_img = 0.5 * gen_img + 0.5

    # Return the generated image as a numpy array
    return gen_img[0, 0].numpy()

    # Return the image
    return gen_img[0, 0].numpy()

# Example: Generate an image with specific labels
ATM_value = 2
CR_value = 1
FR_value = 3
PS_value = 1
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Call the function
generated_image = generate_specific_image(ATM_value, CR_value, FR_value, PS_value, generator, latent_dim, device)

# Visualize the image (optional)
import matplotlib.pyplot as plt
plt.imshow(generated_image, cmap='gray')
plt.title(f"ATM={ATM_value}, CR={CR_value}, FR={FR_value}, PS={PS_value}")
plt.axis('off')
plt.show()

print("Image 1 range:", np.min(image_11), np.max(image_11))