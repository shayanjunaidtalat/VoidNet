import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import os
import torch.optim as optim
from newmodel import UNET
import matplotlib.pyplot as plt
import torch.multiprocessing
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
torch.multiprocessing.set_sharing_strategy('file_system')
from newutils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_best,#!NEW
)

# Hyperparameters etc.

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
# IMAGE_HEIGHT = 160  # 1280 originally
# IMAGE_WIDTH = 240  # 1918 originally
IMAGE_HEIGHT = 112  
IMAGE_WIDTH = 700  
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_BEST = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler,epoch,writer):
    loop = tqdm(loader)
    total_loss_per_epoch = 0
    number_of_images = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        for i in range(len(targets)):
            if torch.max(targets[i]).item() > 1:
                targets[i][targets[i]>=1] = 1

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss_per_epoch += loss.item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        number_of_images += data.shape[0]
        train_fn.training_loss = total_loss_per_epoch/number_of_images
        
        # update tqdm loop
        # loop.set_postfix(loss=loss.item())
        loop.set_postfix(loss=train_fn.training_loss)

        


def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    text = "runs/batchsize" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "loss" + str(loss_fn) + "optim" + optimizer.__class__.__name__ + datetime.now().strftime("%Y_%m_%d%H_%M_%S")
    dir = os.path.join(os.getcwd(),text)
    writer = SummaryWriter(log_dir=dir)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


    #!Additions
    optimizer.zero_grad()
    #!
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    #!NEW
    if LOAD_BEST:
        load_checkpoint(torch.load("best_model.pth.tar"), model)
        print("Below are the stats of current best model")
        maxim = check_accuracy(val_loader, model, loss_fn, writer, device=DEVICE)
        maxim = maxim.item()

    #!NEW
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        print("Below are the stats of last saved model")
        check_accuracy(val_loader, model, loss_fn, writer, device=DEVICE)



    scaler = torch.cuda.amp.GradScaler()
    summary(model, (3, 112, 700),col_names= ("input_size", "kernel_size", "output_size", "num_params", "mult_adds"),depth=6)
    for epoch in range(NUM_EPOCHS):
        #!Returning loss in the train_fn function
        train_fn(train_loader, model, optimizer, loss_fn, scaler,epoch,writer)
        print(f"The Average Training Loss for this epoch is {train_fn.training_loss * 100} %")
        
        #!NEW
        if LOAD_BEST:
            current = check_accuracy(val_loader, model, loss_fn, writer, device=DEVICE)
            current = current.item()
            if current > maxim:
                # save model
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                }
                save_best(checkpoint)
                maxim = current
        #!NEW
        else:
        #! CURRENT
            current = check_accuracy(val_loader, model, loss_fn, writer, device=DEVICE)
            # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        #writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Dice/train", current, epoch)
        writer.add_scalar("Training Loss/train", train_fn.training_loss, epoch)
        writer.add_scalar("Test Loss/test", check_accuracy.test_loss, epoch)

        # # check accuracy
        #check_accuracy(val_loader, model, device=DEVICE)
        #!CURRENT


        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    writer.flush()


if __name__ == "__main__":
    main()
torch.cuda.empty_cache()