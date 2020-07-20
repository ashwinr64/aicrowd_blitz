import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import transformers

from model import BERTBaseUncased, DistilBERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    df_train = pd.read_csv(config.TRAINING_FILE).fillna("none")
    df_train.label = df_train.label.apply(
        lambda x: 1 if x == "unscrambled" else 0)

    df_valid = pd.read_csv(config.VALID_FILE).fillna("none")
    df_valid.label = df_valid.label.apply(
        lambda x: 1 if x == "unscrambled" else 0)

    # df_train, df_valid = model_selection.train_test_split(
    #     dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
    # )

    # df_train = df_train.reset_index(drop=True)
    # df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        text=df_train.text.values, target=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        text=df_valid.text.values, target=df_valid.label.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")
    # model = DistilBERTBaseUncased()
    configuration = transformers.DistilBertConfig()
    # Initializing a model from the configuration
    # model = transformers.DistilBertModel(configuration)
    model = transformers.DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased')
    model.classifier = nn.Linear(768, 1)
    print(model)
    # exit(0)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_f1 = 0
    es_patience = 3
    es = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(
            train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        # valid_loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score = metrics.f1_score(targets, outputs)
        print(f"Accuracy Score = {accuracy} F1 Score = {f1_score}")
        if f1_score > best_f1:
            print(
                f'Saving model, F1 score improved from {best_f1} to {f1_score}')
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_f1 = f1_score
        else:
            if es < es_patience:
                print(f'Early stopping!')
                break
            else:
                es += 1
                print(f'Early Stop Counter {es} of {es_patience}')


if __name__ == "__main__":
    run()
