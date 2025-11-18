import argparse
import os
import torch
import torch.nn as nn
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


###############################################
# DATASET + COLLATE
###############################################

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    masks = [x["attention_mask"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)

    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_masks,
        "labels": labels,
    }


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer):
        self.smiles = list(smiles_list)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


###############################################
# MODEL
###############################################

class PeptideModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate, target, tokenizer):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        dim = self.model.config.embed_dim

        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(dim, 1)

        if target == "classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.SmoothL1Loss()

        self.learning_rate = learning_rate
        self.target = target

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )["mean_pool"]

        x = self.fc1(out)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(-1)

    def training_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


###############################################
# TRAIN FUNCTION
###############################################

def train_model(train_smiles, train_labels, val_smiles, val_labels, target,
                model_name, gpu, batch_size=16, max_epochs=10, learning_rate=3e-4):

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    train_ds = MoleculeDataset(train_smiles, train_labels, tokenizer)
    val_ds = MoleculeDataset(val_smiles, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

    model = PeptideModel(model_name=model_name,
                         learning_rate=learning_rate,
                         target=target,
                         tokenizer=tokenizer)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[gpu],
        val_check_interval=0.2,
        logger=CSVLogger("logs/", name=f"{dataset}_{model_name.split('/')[-1]}"),
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, earlystop_cb],
    )

    trainer.fit(model, train_loader, val_loader)

    # Load the best version
    best_path = checkpoint_cb.best_model_path
    model = PeptideModel.load_from_checkpoint(best_path,
                                              model_name=model_name,
                                              learning_rate=learning_rate,
                                              target=target,
                                              tokenizer=tokenizer)

    return model


###############################################
# TEST / INFERENCE
###############################################

def evaluate_on_test_set(model, test_smiles, test_labels, model_name, batch_size=64):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    test_ds = MoleculeDataset(test_smiles, test_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)

    model.eval()
    model.to("cuda")

    preds = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()

            logits = model(ids, mask)
            preds.extend(logits.cpu().numpy())

    return np.array(preds)


###############################################
# TASK TYPE INFERENCE
###############################################

def infer_task(labels):
    if labels.dtype == object:
        return "classification"
    if labels.nunique() <= 10:
        return "classification"
    return "regression"


###############################################
# MAIN
###############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    dataset = args.dataset
    gpu = args.gpu
    model_name = args.model_name

    train_df = pd.read_csv(f"data/{dataset}_train.csv")
    val_path = f"data/{dataset}_val.csv"
    test_path = f"data/{dataset}_test.csv"

    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

    task = infer_task(train_df["label"])

    ###############################################
    # TRAIN + EVAL PATHS
    ###############################################

    if val_df is not None and test_df is not None:
        model = train_model(train_df["smiles"], train_df["label"],
                            val_df["smiles"], val_df["label"],
                            task, model_name, gpu)

        preds = evaluate_on_test_set(model, test_df["smiles"], test_df["label"], model_name)

        out = pd.DataFrame({
            "smiles": test_df["smiles"],
            "true_label": test_df["label"],
            "predicted_label": preds,
        })

        os.makedirs("results", exist_ok=True)
        out.to_csv(f"results/{dataset}_{model_name.split('/')[-1]}_results.csv", index=False)

    else:
        # fallback to 5-fold CV
        kf = KFold(n_splits=5, shuffle=True)

        all_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            tr = train_df.iloc[train_idx]
            va = train_df.iloc[val_idx]

            model = train_model(tr["smiles"], tr["label"],
                                va["smiles"], va["label"],
                                task, model_name, gpu)

            if test_df is not None:
                preds = evaluate_on_test_set(model, test_df["smiles"], test_df["label"], model_name)

                df = pd.DataFrame({
                    "smiles": test_df["smiles"],
                    "true_label": test_df["label"],
                    "predicted_label": preds,
                    "fold": fold,
                })
                all_results.append(df)
                continue
            elif test_df is None:
                # evaluate on created val set
                preds = evaluate_on_test_set(model, va["smiles"], va["label"], model_name)

            df = pd.DataFrame({
                "smiles": va["smiles"],
                "true_label": va["label"],
                "predicted_label": preds,
                "fold": fold,
            })
            all_results.append(df)

        final = pd.concat(all_results)
        os.makedirs("results", exist_ok=True)
        final.to_csv(f"results/{dataset}_{model_name.split('/')[-1]}_results.csv", index=False)

