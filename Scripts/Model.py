from Feature_engineering import train_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import os



#Calculating class distribution for imbalanced data
def get_class_distribution(dataloader, num_classes):
    print("Calculating class distribution...")
    class_counts = torch.zeros(num_classes)
    total_pixels = 0

    for _, masks in dataloader:
        # Flatten masks to 1D to count occurrences
        flat_masks = masks.view(-1)
        # bincount calculates frequency of each integer
        counts = torch.bincount(flat_masks, minlength=num_classes)
        class_counts += counts
        total_pixels += flat_masks.numel()

    # Convert to percentages
    percentages = (class_counts / total_pixels) * 100

    for i in range(num_classes):
        print(f"Class {i}: {class_counts[i].item():.0f} pixels ({percentages[i].item():.2f}%)")

    return class_counts, percentages

get_class_distribution(train_loader, 4)

##Attention gate for skip connections
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), #1x1 convolution over the lower (deeper) layer
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), #1x1 convolution over  current the encoder layer
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.mean_psi = 0.0

    def forward(self, g, x):
        #g is lower resolution better features signal while x is higher resolution
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        #We match g and x size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True) #Align the width and height.

        psi = self.relu(g1 + x1) #Combine lower and current encoder layers (current layer has higher resolution and lower layer has higher intelligence_
        psi = self.psi(psi) #Run the convution through the sigmoid

        self.mean_psi = psi.mean().item()

        return x * psi #Enriched skip connection we only focus on the important parts not noise.



class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        ) ##standard double convolution to capture deeper features

        self.relu = nn.ReLU(inplace=True) #Relu for non linearity
        self.dropout = nn.Dropout2d(dropout_rate) #Drop out if needed for regularization

        # match channels for residual
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) #Reshape the channels by 1x1 convolution
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x) #Previous stage feature map transformed by 1x1 conv
        out = self.conv(x) #New feature map
        out = out + res #We add them
        out = self.relu(out)
        out = self.dropout(out)
        return out #Return the new feature map. Stabilizes gradient


class BetterUNet(nn.Module):
    def __init__(self, num_classes, f=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ResDoubleConv(3, f) # Size 3x256x256 -> 32x256x256, kernel 32x3x3x3
        self.enc2 = ResDoubleConv(f, f*2) #Size 32x128x128 -> 64x128x128, kernel 64x32x3x3
        self.enc3 = ResDoubleConv(f*2, f*4) #Size 64x64x64 -> 128x64x64, kernel 128x64x3x3
        self.enc4 = ResDoubleConv(f*4, f*8) #Size 128x32x32 -> 256x32x32 kernel 256x128x3x3

        #We lower width and height because of max pooling with stride 2

        # Bridge
        self.bridge = ResDoubleConv(f*8, f*16) #Size 256x16x16 -> 512x16x16 kernel 512x256x3x3

        # Attention Gates
        self.att4 = AttentionGate(F_g=f*16, F_l=f*8, F_int=f*8) #256x32x32
        self.att3 = AttentionGate(F_g=f*8, F_l=f*4, F_int=f*4)  #128x64x64
        self.att2 = AttentionGate(F_g=f*4, F_l=f*2, F_int=f*2)  #64x128x128
        self.att1 = AttentionGate(F_g=f*2, F_l=f,   F_int=f)    #32x256x256

        # Decoder
        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)  #512x16x16 -> #256x32x32
        self.dec4 = ResDoubleConv(f*16, f*8) #512x32x32 -> #256x32x32

        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2) #256x32x32 -> #128x64x64
        self.dec3 = ResDoubleConv(f*8, f*4) #256x64x64-> 128x64x64

        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2) #128x64x64 -> #64x128x128
        self.dec2 = ResDoubleConv(f*4, f*2) #128x128x128 -> 64x128x128

        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2) #64x128x128 -> #32x256x256
        self.dec1 = ResDoubleConv(f*2, f) #64x256x256 -> 32x256x256

        self.out = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bridge
        b = self.bridge(self.pool(s4))

        # Decoder with Attention
        d4 = self.up4(b) #Take the bridge 512x16x16 -> 512x32x32
        s4 = self.att4(g=b, x=s4)  #Run through gate to convert to 256x32x32
        d4 = self.dec4(torch.cat([d4, s4], dim=1)) #Add to 512x32x32

        d3 = self.up3(d4)
        s3 = self.att3(g=d4, x=s3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(g=d3, x=s2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(g=d2, x=s1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.out(d1)

    def get_attention_stats(self):
        #Just checking the are gates active.
        return {
            "Gate_4_Deep": self.att4.mean_psi,
            "Gate_3": self.att3.mean_psi,
            "Gate_2": self.att2.mean_psi,
            "Gate_1_Wide": self.att1.mean_psi
        }


class FocalLoss(nn.Module):
    #Focal loss lowers the impact of loss function by (1-p)^2 so it doesn't give high impact to easy pixels
    def __init__(self, gamma=2, weights=None):
        super().__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, outputs, targets):
        ce = F.cross_entropy(outputs, targets,
                             weight=self.weights, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        #Predicted pixels using softmax
        outputs = F.softmax(outputs, dim=1).float()

        # One-hot encode targets to match output shape (Batch, Classes, H, W)
        num_classes = outputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Calculate intersection and union per class
        dims = (0, 2, 3)
        intersection = torch.sum(outputs * targets_one_hot, dims)
        cardinality = torch.sum(outputs + targets_one_hot, dims)

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth) #Dice very heavially penalizes spacial mistakes
        #Top is the probability of a good class / small number if it gets the class wrong (High loss)
        #It really cares about overlapping.
        # I return 1 - dice because we want to minimize the loss
        return 1 - dice_score.mean()

class ComboLoss(nn.Module):
    def __init__(self, weights=None):
        super(ComboLoss, self).__init__()
        self.focal = FocalLoss(gamma=2, weights=weights)
        self.dice = DiceLoss()
        # Tracking
        self.last_focal = 0.0
        self.last_dice = 0.0

    def forward(self, outputs, targets):
        focal_loss = self.focal(outputs, targets)
        dice_loss = self.dice(outputs, targets)

        self.last_focal = focal_loss.item()
        self.last_dice = dice_loss.item()

        return 0.4 * focal_loss + 0.6 * dice_loss


if __name__ == '__main__':

    #Select if I am training on GPU, M2 chip for MAC or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    #Clear GPU memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    weights = torch.tensor([1.0, 1.5, 1.2,1.5]).to(device)

    #Initialize the model
    model = BetterUNet(num_classes=4).to(device)

    #Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)


    #Loss and dynamic scheduler
    criterion = ComboLoss(weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,threshold=0.01
    )

    #To start training again if needed.
    checkpoint_path = "attention_res_network_turbo.pth"

    if os.path.exists(checkpoint_path):
        print(f"--- Loading Checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore the model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore the epoch number
        start_epoch = checkpoint['epoch']
        print(f"Resuming from Epoch {start_epoch}")
    else:
        print("--- No checkpoint found. Starting from scratch. ---")
        start_epoch = 0

    num_epochs = 50

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Start training the model

        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0

        for i, (images, masks) in enumerate(train_loader):
            # Move data to GPU
            images = images.to(device)
            masks = masks.to(device)

            # Calculate the model outputs
            outputs = model(images)

            #Calculate loss
            loss = criterion(outputs, masks)

            current_ce = criterion.last_focal
            current_dice = criterion.last_dice

            # --- Backward Pass (Optimization) ---
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Calculate new gradients
            optimizer.step()  # Update weights

            #Track loss function
            running_loss += loss.item()
            running_ce += current_ce
            running_dice += current_dice

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}]")
                print(f"  > Total Loss:  {loss.item():.4f}")
                print(f"  > Focal Loss:  {criterion.last_focal:.4f}")
                print(f"  > Dice Loss:   {criterion.last_dice:.4f}")
                print("-" * 30)



        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

        if (epoch + 1) % 5 == 0:
            print(f"\n{'=' * 20} ATTENTION MONITOR {'=' * 20}")
            attn_stats = model.get_attention_stats()
            for gate, val in attn_stats.items():
                status = "Focusing" if val < 0.5 else "Idle/Open"
                print(f"  {gate}: {val:.4f} -> [{status}]")
            print('=' * 55 + '\n')
        #Take 1 batch and calculate the statistics to see if I am not just predicting the background
        if epoch >= 0:
            model.eval()
            with torch.no_grad():
                images, masks = next(iter(train_loader))
                images = images.to(device)

                # Predict on the WHOLE batch
                pred = model(images)
                pred_mask = pred.argmax(dim=1)

                print(f"--- Batch Statistics (Size: {images.size(0)}) ---")
                print("Predicted classes in batch:", pred_mask.unique().tolist())
                print("GT classes in batch:       ", masks.unique().tolist())

                for c in range(4):
                    # Calculate % across the whole batch
                    pct = (pred_mask == c).float().mean().item() * 100
                    print(f"  Class {c}: {pct:.1f}% of total pixels in batch")
            model.train()

        #Checkpoint so that I don't lose the model
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }
        torch.save(checkpoint_data, "attention_res_network_turbo.pth")
        print(f"Saved latest checkpoint to latest_checkpoint.pth")

