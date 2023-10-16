import torch
import torchvision
from dataset import BubbleDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.multiprocessing
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')

def save_best(state, filename="best_model.pth.tar"):
    print("=> Saving Best Model")
    torch.save(state, filename)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = BubbleDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BubbleDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, loss_fn, writer, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        test_loss = 0
        number_of_images = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y[y>=1] = 1
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            test_loss += loss_fn(model(x),y)
            number_of_images += x.shape[0]
    check_accuracy.test_loss = test_loss.item()/number_of_images
    print(f"The Average Test Loss is {check_accuracy.test_loss * 100} %")
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {(dice_score/len(loader)) * 100} %")
    model.train()
    return dice_score/len(loader)

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        #torchvision.utils.save_image(
        #    preds, f"{folder}/pred_{idx}.png"
        #
        #)
        #torchvision.utils.save_image(y.unsqueeze(1), f"{folder}mask_{idx}.png")
        # torchvision.utils.save_image(x[:,0,:,:].unsqueeze(1), f"{folder}/image_{idx}.png")
        input_img = x[:,0,:,:].unsqueeze(1)
        output_img = y.unsqueeze(1)
        prediction_img = preds
        merge = np.concatenate((output_img.detach().cpu().numpy()/1, input_img.detach().cpu().numpy()/1), axis=2)
        merge = np.concatenate((merge, prediction_img.detach().cpu().numpy()/1), axis=2)
        
        for i in range(len(merge)):
            torchvision.utils.save_image(torch.tensor(merge[i][0], device = device), f"{folder}/image_{idx}_{i}.png")
    model.train()