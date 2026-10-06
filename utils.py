from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm_




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_STATE = 73

VAL_SIZE = 0.33

MAX_SEQ_LEN = 64

PAD_IDX = 0
UNK_IDX = 1
BGN_IDX = 2

BATCH_SIZE = 512

D_MODEL = 128

GRU_NUM_LAYERS = 2

DROPOUT = 0.2

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5

EPOCHS = 10

VIDEO_ID_TO_VIDEO_IDX_PATH = "artifacts/video_id_to_video_idx.pickle"
VIDEO_IDX_TO_VIDEO_ID_PATH = "artifacts/video_idx_to_video_id.pickle"
USER_ENCODER_KWARGS_PATH = "artifacts/user_encoder_kwargs.pickle"
VIDEO_ENCODER_KWARGS_PATH = "artifacts/video_encoder_kwargs.pickle"
USER_ENCODER_PATH = "artifacts/user_encoder.pth"
VIDEO_ENCODER_PATH = "artifacts/video_encoder.pth"
ENCODERS_TRAINER_PATH = "artifacts/encoders_trainer.pth"
VIDEO_EMBEDDINGS_INDEX_PATH = "artifacts/video_embeddings.index"



class DatasetForTraining(Dataset):
    """
    Dataset for training.
    """


    def __init__(self,
        interactions: pd.DataFrame,
        video_idx_probas: dict[int, float],
        unk_video_idx_proba: float = 0.0,
        num_neg_video_idxs: int = 1024,
        random_state: int = 73) -> None:
        """
        Initialize the dataset.

        Args:
            interactions (pd.DataFrame): The interactions dataframe.
            video_idx_probas (dict[int, float]): The video index probabilities.
            unk_video_idx_proba (optional, float): The unknown video index probability.
                Default is 0.0.
            num_neg_video_idxs (optional, int): The number of negative video indices.
                Default is 2048.
            random_state (optional, int): The random state.
                Default is 73.
        """

        self.interactions = interactions

        self.video_idx_probas = video_idx_probas
        self.video_idxs = np.array(list(video_idx_probas.keys()))
        self.video_probas = np.array(list(video_idx_probas.values()))
        self.unk_video_idx_proba = unk_video_idx_proba

        self.num_neg_video_idxs = num_neg_video_idxs

        self.rng = np.random.default_rng(random_state)


    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """

        return self.interactions.shape[0]


    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict[str, Any]: The item at the given index.
        """

        item = {}

        item["user_id"] = int(self.interactions['USER_ID'].iloc[idx])
        item["timestamp"] = self.interactions['TIMESTAMP'].iloc[idx]

        item["video_idx_seq"] = self.interactions['VIDEO_IDX_SEQ'].iloc[idx]
        item["video_idx_seq_len"] = int(self.interactions['VIDEO_IDX_SEQ_LEN'].iloc[idx])

        pos_video_idx = int(self.interactions['VIDEO_IDX'].iloc[idx])
        pos_video_proba = self.video_idx_probas.get(pos_video_idx, self.unk_video_idx_proba)

        item["pos_video_idx"] = pos_video_idx
        item["pos_video_proba"] = pos_video_proba

        neg_video_idxs = self.rng.choice(
            a=self.video_idxs,
            size=self.num_neg_video_idxs,
            replace=False,
            p=self.video_probas
        )

        mask = neg_video_idxs == pos_video_idx

        while mask.any():

            neg_video_idxs[mask] = self.rng.choice(
                a=self.video_idxs,
                size=int(mask.sum()),
                replace=False,
                p=self.video_probas
            )

            mask = neg_video_idxs == pos_video_idx

        neg_video_idxs = neg_video_idxs.tolist()
        neg_video_probas = [
            self.video_idx_probas[neg_video_idx]
            for neg_video_idx in neg_video_idxs
        ]

        item["neg_video_idxs"] = neg_video_idxs
        item["neg_video_probas"] = neg_video_probas

        return item



class DatasetForEvaluation(Dataset):
    """
    Dataset for evaluation.
    """


    def __init__(self,
        interactions: pd.DataFrame) -> None:
        """
        Initialize the dataset.

        Args:
            interactions (pd.DataFrame): The interactions dataframe.
        """

        self.interactions = interactions


    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """

        return self.interactions.shape[0]


    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict[str, Any]: The item at the given index.
        """

        item = {}

        item["user_id"] = int(self.interactions['USER_ID'].iloc[idx])
        item["timestamp"] = self.interactions['TIMESTAMP'].iloc[idx]

        item["video_idx_seq"] = self.interactions['VIDEO_IDX_SEQ'].iloc[idx]
        item["video_idx_seq_len"] = int(self.interactions['VIDEO_IDX_SEQ_LEN'].iloc[idx])

        return item



def training_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate function for the training dataset.

    Args:
        batch (list[dict[str, Any]]): The batch of items.

    Returns:
        dict[str, torch.Tensor]: The collated batch.
    """

    video_idx_seq = [
        torch.tensor(item["video_idx_seq"], dtype=torch.long)
        for item in batch
    ]

    video_idx_seq = pad_sequence(
        sequences=video_idx_seq,
        batch_first=True,
        padding_value=PAD_IDX
    )

    video_idx_seq_len = torch.tensor(
        [item["video_idx_seq_len"] for item in batch],
        dtype=torch.long
    )

    pos_video_idx = torch.tensor(
        [item["pos_video_idx"] for item in batch],
        dtype=torch.long
    )

    pos_video_proba = torch.tensor(
        [item["pos_video_proba"] for item in batch],
        dtype=torch.float
    )

    neg_video_idxs = torch.tensor(
        [item["neg_video_idxs"] for item in batch],
        dtype=torch.long
    )

    neg_video_probas = torch.tensor(
        [item["neg_video_probas"] for item in batch],
        dtype=torch.float
    )

    return {
        "video_idx_seq": video_idx_seq,
        "video_idx_seq_len": video_idx_seq_len,

        "pos_video_idx": pos_video_idx,
        "pos_video_proba": pos_video_proba,

        "neg_video_idxs": neg_video_idxs,
        "neg_video_probas": neg_video_probas
    }



def evaluation_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate function for the evaluation dataset.

    Args:
        batch (list[dict[str, Any]]): The batch of items.

    Returns:
        dict[str, torch.Tensor]: The collated batch.
    """

    video_idx_seq = [
        torch.tensor(item["video_idx_seq"], dtype=torch.long)
        for item in batch
    ]

    video_idx_seq = pad_sequence(
        sequences=video_idx_seq,
        batch_first=True,
        padding_value=PAD_IDX
    )

    video_idx_seq_len = torch.tensor(
        [item["video_idx_seq_len"] for item in batch],
        dtype=torch.long
    )

    return {
        "video_idx_seq": video_idx_seq,
        "video_idx_seq_len": video_idx_seq_len
    }



class UserEncoder(nn.Module):
    """
    User encoder.
    """


    def __init__(self,
        videos_crd: int,
        d_model: int,
        pad_idx: int,
        gru_num_layers: int,
        dropout: float) -> None:
        """
        Initialize the user encoder.

        Args:
            videos_crd (int): The number of video indices.
            d_model (int): The dimension of the model.
            pad_idx (int): The padding index.
            gru_num_layers (int): The number of GRU layers.
            dropout (float): The dropout rate.
        """

        super(UserEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=videos_crd,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(p=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=d_model,
                out_features=d_model
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=d_model,
                out_features=d_model
            )
        )


    def forward(self,
        video_idx_seq: torch.Tensor,
        video_idx_seq_len: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video_idx_seq (torch.Tensor): The video index sequence.
            video_idx_seq_len (torch.Tensor): The length of the video index sequence.

        Returns:
            torch.Tensor: The output of the user encoder.
        """

        video_emb_seq = self.embedding(video_idx_seq)

        video_emb_seq_packed = pack_padded_sequence(
            input=video_emb_seq,
            lengths=video_idx_seq_len.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )

        _, video_emb_seq_hidden = self.gru(video_emb_seq_packed)

        video_emb_seq_hidden_last = video_emb_seq_hidden[-1]

        video_emb_seq_hidden_last = self.dropout(video_emb_seq_hidden_last)

        output = self.mlp(video_emb_seq_hidden_last)

        return output



class VideoEncoder(nn.Module):
    """
    Video encoder.
    """


    def __init__(self,
        videos_crd: int,
        d_model: int,
        pad_idx: int) -> None:
        """
        Initialize the video encoder.

        Args:
            videos_crd (int): The number of video indices.
            d_model (int): The dimension of the model.
            pad_idx (int): The padding index.
        """

        super(VideoEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=videos_crd,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )


    def forward(self, video_idxs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video_idxs (torch.Tensor): The video indices.

        Returns:
            torch.Tensor: The output of the video encoder.
        """

        output = self.embedding(video_idxs)

        return output



class EncodersTrainer(nn.Module):
    """
    Encoders trainer.
    """


    def __init__(self,
        user_encoder: UserEncoder,
        video_encoder: VideoEncoder) -> None:
        """
        Initialize the encoders trainer.

        Args:
            user_encoder (UserEncoder): The user encoder.
            video_encoder (VideoEncoder): The video encoder.
        """

        super(EncodersTrainer, self).__init__()

        self.user_encoder = user_encoder
        self.video_encoder = video_encoder


    def forward(self,
        video_idx_seq: torch.Tensor,
        video_idx_seq_len: torch.Tensor,
        pos_video_idx: torch.Tensor,
        pos_video_proba: torch.Tensor,
        neg_video_idxs: torch.Tensor,
        neg_video_probas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video_idx_seq (torch.Tensor): The video index sequence.
            video_idx_seq_len (torch.Tensor): The length of the video index sequence.
            pos_video_idx (torch.Tensor): The positive video index.
            pos_video_proba (torch.Tensor): The positive video probability.
            neg_video_idxs (torch.Tensor): The negative video indices.
            neg_video_probas (torch.Tensor): The negative video probabilities.

        Returns:
            torch.Tensor: The output of the encoders trainer.
        """

        video_idxs = torch.cat([pos_video_idx.unsqueeze(-1), neg_video_idxs], dim=1)
        video_probas = torch.cat([pos_video_proba.unsqueeze(-1), neg_video_probas], dim=1)

        user_embedding = self.user_encoder(
            video_idx_seq,
            video_idx_seq_len
        )

        video_embedding = self.video_encoder(
            video_idxs
        )

        logits = torch.matmul(
            user_embedding.unsqueeze(1),
            video_embedding.transpose(1, 2)
        ).squeeze(1)

        output = logits - torch.log(video_probas.clamp(min=1e-12))

        return output



def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device) -> dict[str, float]:
    """
    Performs a training step.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The dataloader of the training data.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on.

    Returns:
        dict[str, float]: The metrics of the training step.
    """

    model = model.to(device)

    train_metrics = {}

    train_loss = 0.0

    model.train()

    for batch in dataloader:

        video_idx_seq = batch["video_idx_seq"].to(device, non_blocking=True)
        video_idx_seq_len = batch["video_idx_seq_len"].to(device, non_blocking=True)

        pos_video_idx = batch["pos_video_idx"].to(device, non_blocking=True)
        pos_video_proba = batch["pos_video_proba"].to(device, non_blocking=True)

        neg_video_idxs = batch["neg_video_idxs"].to(device, non_blocking=True)
        neg_video_probas = batch["neg_video_probas"].to(device, non_blocking=True)

        logits = model(
            video_idx_seq,
            video_idx_seq_len,
            pos_video_idx,
            pos_video_proba,
            neg_video_idxs,
            neg_video_probas
        )

        targets = torch.zeros(
            logits.shape[0],
            dtype=torch.long,
            device=device
        )

        loss = criterion(logits, targets)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    train_loss /= len(dataloader)

    train_metrics["train_loss"] = train_loss

    train_hit_10_count = 0
    train_hit_100_count = 0
    train_hit_1000_count = 0

    train_samples_count = 0

    model.eval()

    with torch.inference_mode():

        for batch in dataloader:

            video_idx_seq = batch["video_idx_seq"].to(device, non_blocking=True)
            video_idx_seq_len = batch["video_idx_seq_len"].to(device, non_blocking=True)

            pos_video_idx = batch["pos_video_idx"].to(device, non_blocking=True)

            user_embedding = model.user_encoder(
                video_idx_seq,
                video_idx_seq_len
            )

            logits = torch.matmul(
                user_embedding,
                model.video_encoder.embedding.weight.transpose(0, 1)
            )

            train_top_1000 = logits.topk(
                k=1000,
                dim=1
            ).indices

            train_hit_10_count += (
                train_top_1000[:, :10] == pos_video_idx.unsqueeze(1)
            ).any(dim=1).sum().item()

            train_hit_100_count += (
                train_top_1000[:, :100] == pos_video_idx.unsqueeze(1)
            ).any(dim=1).sum().item()

            train_hit_1000_count += (
                train_top_1000 == pos_video_idx.unsqueeze(1)
            ).any(dim=1).sum().item()

            train_samples_count += logits.shape[0]

        train_hit_10_rate = train_hit_10_count / train_samples_count
        train_hit_100_rate = train_hit_100_count / train_samples_count
        train_hit_1000_rate = train_hit_1000_count / train_samples_count

        train_metrics["train_hit_10_rate"] = train_hit_10_rate
        train_metrics["train_hit_100_rate"] = train_hit_100_rate
        train_metrics["train_hit_1000_rate"] = train_hit_1000_rate

    return train_metrics



def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device) -> dict[str, float]:
    """
    Performs a validation step.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): The dataloader of the validation data.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to validate on.

    Returns:
        dict[str, float]: The metrics of the validation step.
    """

    model = model.to(device)

    val_metrics = {}

    val_loss = 0.0

    val_hit_10_count = 0
    val_hit_100_count = 0
    val_hit_1000_count = 0

    val_samples_count = 0

    model.eval()

    with torch.inference_mode():

        for batch in dataloader:

            video_idx_seq = batch["video_idx_seq"].to(device, non_blocking=True)
            video_idx_seq_len = batch["video_idx_seq_len"].to(device, non_blocking=True)

            pos_video_idx = batch["pos_video_idx"].to(device, non_blocking=True)
            pos_video_proba = batch["pos_video_proba"].to(device, non_blocking=True)

            neg_video_idxs = batch["neg_video_idxs"].to(device, non_blocking=True)
            neg_video_probas = batch["neg_video_probas"].to(device, non_blocking=True)

            logits = model(
                video_idx_seq,
                video_idx_seq_len,
                pos_video_idx,
                pos_video_proba,
                neg_video_idxs,
                neg_video_probas
            )

            targets = torch.zeros(
                logits.shape[0],
                dtype=torch.long,
                device=device
            )

            loss = criterion(logits, targets)

            val_loss += loss.item()

            user_embedding = model.user_encoder(
                video_idx_seq,
                video_idx_seq_len
            )

            logits = torch.matmul(
                user_embedding,
                model.video_encoder.embedding.weight.transpose(0, 1)
            )

            val_top_1000 = logits.topk(
                k=1000,
                dim=1
            ).indices

            val_hit_10_count += (
                val_top_1000[:, :10] == pos_video_idx.unsqueeze(1)
            ).any(dim=1).sum().item()

            val_hit_100_count += (
                val_top_1000[:, :100] == pos_video_idx.unsqueeze(1)
            ).any(dim=1).sum().item()

            val_hit_1000_count += (
                val_top_1000 == pos_video_idx.unsqueeze(1)
            ).any(dim=1).sum().item()

            val_samples_count += logits.shape[0]

        val_loss /= len(dataloader)

        val_metrics["val_loss"] = val_loss

        val_hit_10_rate = val_hit_10_count / val_samples_count
        val_hit_100_rate = val_hit_100_count / val_samples_count
        val_hit_1000_rate = val_hit_1000_count / val_samples_count

        val_metrics["val_hit_10_rate"] = val_hit_10_rate
        val_metrics["val_hit_100_rate"] = val_hit_100_rate
        val_metrics["val_hit_1000_rate"] = val_hit_1000_rate

    return val_metrics