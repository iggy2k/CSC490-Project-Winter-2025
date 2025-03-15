import torch
from tqdm import tqdm
from livelossplot import PlotLosses

def train_model(model, optimizer, train_dataloader, accelerator, scheduler, num_epochs=1, haversineLoss=None, OUTPUT_CONTEXT=None):
    """
    Trains the given model using the specified optimizer and data loader.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_dataloader (torch.utils.data.DataLoader): The training data loader.
        accelerator (accelerate.Accelerator): The accelerator for distributed training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train for.
        haversineLoss (torch.nn.Module): Loss function for Haversine distance.
        OUTPUT_CONTEXT (ipywidgets.Output, optional): Output context for logging losses.

    Returns:
        list: Average loss per epoch.
    """
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    avg_epoch_loss = []
    plotlosses = PlotLosses(figsize=(10, 5))

    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Epochs"):
        train_loss = []
        
        for images, coordinates, image_path in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1} progress"):
            if torch.cuda.is_available():
                images, coordinates = images.cuda(), coordinates.cuda()

            optimizer.zero_grad()
            outputs = model(images, coordinates, image_path)

            loss_haversine = haversineLoss(outputs["pred"], coordinates).mean(dim=-1)

            pbar_msg = f'Loss: {loss_haversine:.5f}; \
                        \nOut: {outputs["pred"].mean(dim=0).tolist()}\
                        \nExpected: {coordinates.mean(dim=0).tolist()}'
            tqdm.write(pbar_msg)
            
            train_loss.append(loss_haversine.detach())

            if OUTPUT_CONTEXT:
                with OUTPUT_CONTEXT:
                    plotlosses.update({
                        "Haversine": loss_haversine.item(),
                        "Cross Entropy": outputs["loss"].detach().item()
                    })
                    plotlosses.send()

            loss_haversine.requires_grad = True
            accelerator.backward(outputs["loss"])
            accelerator.backward(loss_haversine)
            optimizer.step()

        scheduler.step()
        avg_loss = sum(train_loss) / len(train_dataloader)
        avg_epoch_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return avg_epoch_loss
