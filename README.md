# Sequential video recommendation with a GRU

Next-video prediction from a user's recent interactions. A GRU encodes the history. A video embedding table is the catalog. Retrieval is inner product over that catalog.

## Data

Place two headerless CSVs:

```
data/train.csv
data/evaluate.csv
```

Columns, in order: `USER_ID`, `VIDEO_ID`, `INTERACTION_TYPE`, `TIMESTAMP` (Unix seconds, UTC). Rows with interaction types `1`, `2`, `3`, and `5` are kept.

`train.csv` covers 2016-11-06 through 2017-01-26 04:59:55. `evaluate.csv` starts at 2017-01-26 05:00:10 and runs through 2017-02-07. The two files do not overlap in time.

Both files and everything under `artifacts/` stay out of git. Empty directories are kept with `.gitkeep`.

## Protocol

Users in `train.csv` are shuffled with seed `73` and split 67/33 (`VAL_SIZE = 0.33`). The vocabulary is built from the training users only. Validation videos missing from that vocabulary map to `UNK`.

Each example is the videos a user watched before the current one, truncated to the last 64. An empty history is a single `BGN` token. Special indices are `PAD = 0`, `UNK = 1`, `BGN = 2`.

The model does not see `USER_ID`. Validation users are unseen, but their full in-file history is available, so train and validation hit rates stay close.

`evaluate.ipynb` scores `evaluate.csv` with the same history rule: previous videos of that user inside the file, truncated to 64.

## Model

Two towers, each with its own embedding table of shape `videos_crd × 128`. `videos_crd` is the number of training video ids plus the three special tokens (`335024` on this split). `PAD` is the padding index in both tables.

`UserEncoder` maps a history to one vector:

1. Embed each video index.
2. Pack the batch by true length and run a 2-layer GRU (`input_size = hidden_size = 128`, dropout `0.2` between layers).
3. Take the final hidden state of the last layer.
4. Dropout (`0.2`), then `Linear(128, 128)`, ReLU, dropout (`0.2`), `Linear(128, 128)`.

`VideoEncoder` is the second embedding table. The score of a candidate is the inner product of the user vector and that video's row.

On this vocabulary the user encoder has `43,114,240` parameters, the video encoder `42,883,072`, and the two together `85,997,312`. The embedding tables hold nearly all of them.

Training uses sampled softmax: 1024 frequency-weighted negatives, a logQ correction, and cross-entropy. Optimization is Adam (`lr = 1e-3`, `weight_decay = 1e-5`) for 10 epochs, batch size 512, gradient clip 1.0. Hit rate is measured with full-catalog top-k, not with the sampled negatives.

After training, video embeddings from index 3 upward (the real catalog, without `PAD`, `UNK`, and `BGN`) are written to a FAISS `IndexFlatIP`. Evaluation searches that index.

Constants live in `utils.py`.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run `train.ipynb`, then `evaluate.ipynb`. Training writes encoder weights, id maps, and `artifacts/video_embeddings.index`. Evaluation reads those files.

On Windows, notebook dataloaders use `num_workers=4`. If worker startup stalls while pickling the dataframe, set `num_workers=0` and drop `persistent_workers`.

## Results

Hit rate after 10 epochs, full catalog:

| Split | Hit@10 | Hit@100 | Hit@1000 |
|---|---:|---:|---:|
| Train | 0.48 | 0.59 | 0.73 |
| Validation | 0.45 | 0.52 | 0.61 |
| Evaluate | 0.47 | 0.57 | 0.69 |

Validation searches the full embedding matrix, including the `UNK` row, so an unknown target can count as a hit. The FAISS index used at evaluation starts at index 3, so unknown targets (about 7% of both validation and evaluation rows) cannot be retrieved.
