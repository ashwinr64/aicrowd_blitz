import pretrainedmodels
from tqdm import tqdm
import random
from PIL import Image, ImageFile, ImageOps, ImageEnhance
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import albumentations
import pandas as pd
import numpy as np
from sklearn import metrics
import time
import sys
import gc


ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

try:
    from apex import amp

    _apex_available = True
except ImportError:
    _apex_available = False

# ---------------- Net ------------------#


class ResNet18(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(ResNet18, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "resnet18"
        ](pretrained=pretrained)
        self.out = nn.Linear(512, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.MSELoss()(
            out, targets.reshape(-1, 1).type_as(out)
        )
        return out, loss


class ResNet50(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(ResNet50, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "resnet50"
        ](pretrained=pretrained)

        self.out = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.MSELoss()(
            out, targets.reshape(-1, 1).type_as(out)
        )
        return out, loss

# ---------------- Engine ------------------#


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        epoch,
        scheduler=None,
        accumulation_steps=1,
        use_tpu=False,
        fp16=False,
    ):
        if use_tpu and not _xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )
        if fp16 and not _apex_available:
            raise Exception(
                "You want to use fp16 but you dont have apex installed")
        if fp16 and use_tpu:
            raise Exception("Apex fp16 is not available when using TPUs")
        if fp16:
            accumulation_steps = 1
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            _, loss = model(**data)

            if not use_tpu:
                with torch.set_grad_enabled(True):
                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    if (b_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        if b_idx > 0:
                            optimizer.zero_grad()
            else:
                loss.backward()
                xm.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                if b_idx > 0:
                    optimizer.zero_grad()

            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)

        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device, epoch, use_tpu=False):
        losses = AverageMeter()
        final_predictions = []
        model.eval()
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, loss = model(**data)

                predictions = predictions.cpu()
                losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)

        return final_predictions, losses.avg

    @staticmethod
    def predict(data_loader, model, device, use_tpu=False):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, _ = model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions

# ---------------- Early Stopping ------------------#


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, epoch, optimizer, model_path, amp=None):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, epoch,
                                 optimizer, model_path, amp)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, epoch,
                                 optimizer, model_path, amp)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, epoch, optimizer, model_path, amp):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            if amp:
                ckpt_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'amp': amp.state_dict()
                }
            else:
                ckpt_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            torch.save(ckpt_dict, model_path)
        self.val_score = epoch_score


class RegressionLoader:
    def __init__(self, image_paths, targets, resize, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        targets = self.targets[item]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image),
            # "image": image,
            "targets": torch.tensor(targets),
        }


def train(fold):
    training_data_path = '../data/training_masked'
    df = pd.read_csv('../data/labels_folds_new.csv')
    device = 'cuda'
    epochs = 50
    train_bs = 32
    valid_bs = 32
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    fp16 = False
    size = 224

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    print(df_train.xRot.value_counts())

    train_aug = albumentations.Compose(
        [
            albumentations.Resize(size, size),
            albumentations.Cutout(
                num_holes=5, max_h_size=30, max_w_size=30, fill_value=255, always_apply=False, p=0.5),
            albumentations.Normalize(),
        ])

    valid_aug = albumentations.Compose(
        [
            albumentations.Resize(size, size),
            albumentations.Normalize(),
        ])

    train_images = df_train.filename.values.tolist()
    train_images = [os.path.join(training_data_path, i)
                    for i in train_images]
    train_targets = df_train.xRot.values

    valid_images = df_valid.filename.values.tolist()
    valid_images = [os.path.join(training_data_path, i)
                    for i in valid_images]
    valid_targets = df_valid.xRot.values

    train_dataset = RegressionLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4,
    )

    valid_dataset = RegressionLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = ResNet18()

    print(model)

    model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode='min'
    )

    # Initialization
    if fp16:
        opt_level = 'O1'
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=opt_level)

    es = EarlyStopping(patience=5, mode='min')

    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device,
            epoch,
            fp16=fp16
        )

        gc.collect()

        predictions, valid_loss = Engine.evaluate(
            valid_loader,
            model,
            device,
            epoch
        )

        predictions = np.vstack((predictions)).ravel()
        print(predictions.shape)

        # For ReduceLROnPlateau
        scheduler.step(valid_loss)

        print(
            f'epoch: {epoch}, train_loss: {training_loss}, valid_loss: {valid_loss}')

        if fp16:
            pass
        else:
            es(valid_loss, model, epoch, optimizer,
               model_path=os.path.join('models', 'model.tar'))

        if es.early_stop:
            print('Early Stopping')
            break


if __name__ == '__main__':
    fold = int(sys.argv[1])
    train(fold)
