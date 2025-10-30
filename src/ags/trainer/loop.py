import torch
from tqdm import tqdm

class Trainer:
    def  __init__(self, model, optimizer, loss_fn, scaler=None, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.scheduler = scheduler
        self.device = device

    def fit(self, train_loader, max_epoch=10, val_fn=None):
        state = {
            "model": self.model, "optimizer": self.optimizer,
            "loss_fn": self.loss_fn, "scaler": self.scaler,
            "device": self.device, "global_step": 0, "max_epoch": max_epoch
        }

        for epoch in tqdm(range(max_epoch)):
            state["epoch"] = epoch
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                state["batch"] = batch

                x, y = (t.to(self.device) for t in batch)
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)

                state["loss"] = loss
                (self.scaler.scale(loss) if self.scaler else loss).backward()

                if self.scaler:
                    self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                state["global_step"] += 1

            self.scheduler.step()
            state["epoch_loss"] = epoch_loss / max(1, len(train_loader))
            if val_fn:
                state.update(val_fn(state))
        print(state["epoch_loss"])
