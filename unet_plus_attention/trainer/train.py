import torch
import gc
from tqdm import tqdm_notebook as tqdm
import numpy as np
from matplotlib import pyplot as plt

from IPython.display import clear_output


def train(
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        val_dataloader,
        scheduler,
        device,
        num_epochs,
        type_=torch.long,
        verbose=False
):
    train_loss_hist = []
    val_loss_hist = []
    epoch_train_loss_hist = []
    epoch_val_loss_hist = []
    model.to(device)
    gc.collect()
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    for e in range(num_epochs):
        print(f'Epoch {e + 1} out of {num_epochs}')
        optimizer.zero_grad()
        model.train(True)
        if verbose:
            train_range = tqdm(train_dataloader)
        else:
            train_range = train_dataloader
        for images, masks in train_range:
            optimizer.zero_grad()
            images = images.to(device)
            masks = masks.to(device).type(type_)
            if type(loss_fn) is torch.nn.modules.loss.CrossEntropyLoss:
                masks = masks.squeeze(1)
            pred = model(images)
            loss = loss_fn(pred, masks)
            loss.backward()
            optimizer.step()
            train_loss_hist.append(loss.detach().cpu().item())
            del loss
            del images
            del masks
            torch.cuda.empty_cache()

        epoch_train_loss_hist.append(np.mean(train_loss_hist[-len(train_dataloader):]))
        scheduler.step()

        gc.collect()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        model.train(False)
        with torch.no_grad():
            if verbose:
                val_range = tqdm(val_dataloader)
            else:
                val_range = val_dataloader
            for images, masks in val_range:
                images = images.to(device)
                masks = masks.to(device).type(type_)
                if type(loss_fn) is torch.nn.modules.loss.CrossEntropyLoss:
                    masks = masks.squeeze(1)

                pred = model(images)
                loss = loss_fn(pred, masks)
                val_loss_hist.append(loss.detach().cpu().item())
                del loss
                del images
                del masks
                torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()
        epoch_val_loss_hist.append(np.mean(val_loss_hist[-len(val_dataloader):]))
        if verbose:
            clear_output()
            print(f'Train: {epoch_train_loss_hist[-1]}')
            print(f'Validation: {epoch_val_loss_hist[-1]}')
            fig, axs = plt.subplots(2, 3, figsize=(25, 15), sharey=False)
            axs[0, 0].plot(np.arange(len(train_loss_hist)), train_loss_hist, label='train')
            axs[0, 0].plot(np.arange(len(val_loss_hist)), val_loss_hist, label='val')
            axs[0, 2].plot(np.arange(len(epoch_train_loss_hist)), epoch_train_loss_hist, label='train')
            axs[0, 2].plot(np.arange(len(epoch_val_loss_hist)), epoch_val_loss_hist, label='val')
            with torch.no_grad():
                batch = next(iter(val_dataloader))

                images = batch[0]
                masks = batch[1]
                image = images[np.random.randint(0, images.shape[0])]
                mask = masks[np.random.randint(0, masks.shape[0])]

                pred_mask = model(image.to(device).unsqueeze(0)).squeeze(0).cpu()

                plt.axis('off')
                axs[1, 0].imshow(image.numpy().transpose(1, 2, 0))
                axs[1, 1].imshow(mask.squeeze(0).numpy())
                axs[1, 2].imshow(pred_mask.argmax(0).detach().squeeze(0).numpy())
                del pred_mask
                gc.collect()
                torch.cuda.empty_cache()

            plt.legend()
            plt.show()
    return train_loss_hist, val_loss_hist
