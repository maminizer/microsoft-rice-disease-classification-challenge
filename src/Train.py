# ====================================================
# Library
# ====================================================
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import CFG
from Utils import LOGGER, OUTPUT_DIR, get_score
from Dataset import TrainDataset, get_transforms, DataLoader
from Model import CustomModel
from Engine import train_fn, valid_fn

train = pd.read_csv('train_df')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERSION = 1

# ====================================================
# Train loop
# ====================================================


def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['folds'] != fold].index
    val_idx = folds[folds['folds'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds["Label"].values

    train_dataset = TrainDataset(
        train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(
        valid_folds, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, **CFG.reduce_params)
        elif CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, **CFG.cosanneal_params)
        elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, **CFG.reduce_params)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr,
                     weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model,
                            criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring

        #preds_label = np.argmax(preds, axis=1)
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(
                f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds_loss': preds},
                       OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth')

    valid_folds[CFG.preds_col] = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth',
                                            map_location=torch.device('cpu'))['preds_loss']

    return valid_folds


# ====================================================
# main
# ====================================================
def main():
    """
    Prepare: 1.train 
    """

    def get_result(result_df):
        preds_loss = result_df[CFG.preds_col].values
        labels = result_df["Label"].values
        score_loss = get_score(labels, preds_loss)
        LOGGER.info(f'Score with best loss weights: {score_loss:<.4f}')

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.nfolds):
            if fold in CFG.trn_folds:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df[['Image_id', 'blast', 'brown', 'healthy']].to_csv(
            OUTPUT_DIR+f'{CFG.model_name}_oof_rgb_df_version{VERSION}.csv', index=False)


if __name__ == "__main__":
    main()
