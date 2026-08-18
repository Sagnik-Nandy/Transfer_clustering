"""GDEC (YuzhiSun/GDEC, cloned into External_Methods/GDEC), GCN-free
version: SDAE pretrain-on-source, weight-only transfer to a fresh target
autoencoder, fine-tune on target, DEC self-training clustering. This
reuses GDEC's actual ptsdae/ptdec modules (StackedDenoisingAutoEncoder,
DEC, the pretrain/train/predict functions) via direct import -- these are
data-agnostic PyTorch building blocks, unrelated to the GCN gene-graph
code we're dropping. We write our own minimal in-memory Dataset wrapper
instead of their file-based, gene-network-embedding-dependent
GeneDotGCNDataSet/GeneNoGCNDataSet classes, since those still require CSV
files and (unused, for the no-GCN case) gene-network paths on disk.

No native multi-source mechanism in the original method (it's 1-source +
1-target) -- we pool the 3 sources into a single combined dataset, the
same simplification used for scRNA and (before it was dropped) TGMM.

Epoch counts match GDEC's own showcased defaults (200-300); early stopping
(see below) keeps this from being wasteful when a stage converges sooner.

Requires `torch` in the `transfer_clustering` conda env (not installed by
default):
    pip install torch
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "External_Methods", "GDEC"))

from common import K

GDEC_HIDDEN_DIMS = [500, 500, 2000, 20]
GDEC_PRETRAIN_EPOCHS = 200   # layer-wise SDAE pretraining (ptsdae.model.pretrain's own
                             # per-layer loop -- see _gdec_gcnfree_impl's note on why this
                             # one stage does not get the early-stopping wrapper below)
GDEC_FINETUNE_EPOCHS = 200   # whole-network fine-tune on source, early-stopped below
GDEC_TRANSFER_EPOCHS = 200   # whole-network fine-tune on target, early-stopped below
GDEC_DEC_EPOCHS = 200        # DEC self-training, early-stopped via ptdec's native stopping_delta
GDEC_FINETUNE_PATIENCE = 15  # epochs of no validation-loss improvement before stopping
GDEC_TRANSFER_PATIENCE = 15
GDEC_DEC_STOPPING_DELTA = 0.001  # fraction of points changing cluster assignment between
                                 # epochs below which DEC self-training is considered
                                 # converged -- matches the original DEC paper's own tol
GDEC_BATCH_SIZE = 256


class _EarlyStop(Exception):
    """Sentinel raised from a ptsdae.model.train update_callback to unwind its
    epoch loop early -- that loop has no native stopping hook, unlike
    ptdec.model.train's stopping_delta (used for the DEC stage below)."""


class _ValLossEarlyStopper:
    """update_callback for ptsdae.model.train (called every epoch, since it's
    invoked with update_freq=1): raises _EarlyStop once `patience` consecutive
    epochs pass with no validation-loss improvement. Only safe to attach to a
    single whole-network train() call (finetune/transfer below) -- NOT to
    ae.pretrain(), whose internal loop trains one sub-autoencoder layer per
    call to train(); raising there would unwind out of pretrain() entirely
    and skip pretraining the remaining layers."""

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.n_bad = 0

    def __call__(self, epoch, lr, loss, validation_loss):
        if validation_loss < self.best - self.min_delta:
            self.best = validation_loss
            self.n_bad = 0
        else:
            self.n_bad += 1
            if self.n_bad >= self.patience:
                raise _EarlyStop


def _gdec_gcnfree_impl(X_T: np.ndarray, X_S: np.ndarray, seed: int) -> np.ndarray:
    import torch
    from torch.optim import SGD
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import Dataset
    from ptsdae.sdae import StackedDenoisingAutoEncoder
    import ptsdae.model as ae
    from ptdec.dec import DEC
    from ptdec.model import train as dec_train, predict as dec_predict

    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()

    class ArrayDataset(Dataset):
        def __init__(self, X):
            self.X = X.astype(np.float32)

        def __getitem__(self, index):
            return torch.tensor(self.X[index]), torch.tensor(0.0)

        def __len__(self):
            return self.X.shape[0]

    d = X_S.shape[1]
    dims = [d] + GDEC_HIDDEN_DIMS
    ds_src = ArrayDataset(X_S)
    ds_trg = ArrayDataset(X_T)

    autoencoder = StackedDenoisingAutoEncoder(dims, final_activation=None)
    if cuda:
        autoencoder.cuda()
    ae.pretrain(
        ds_src, autoencoder, cuda=cuda, validation=ds_src, epochs=GDEC_PRETRAIN_EPOCHS,
        batch_size=GDEC_BATCH_SIZE, optimizer=lambda m: SGD(m.parameters(), lr=0.5, momentum=0.9),
        scheduler=lambda o: StepLR(o, 30, gamma=0.1), corruption=0.2, silent=True,
    )
    ae_optimizer = SGD(autoencoder.parameters(), lr=0.5, momentum=0.9)
    try:
        ae.train(
            ds_src, autoencoder, cuda=cuda, validation=ds_src, epochs=GDEC_FINETUNE_EPOCHS,
            batch_size=GDEC_BATCH_SIZE, optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer, 30, gamma=0.1), corruption=0.2, silent=True,
            update_callback=_ValLossEarlyStopper(GDEC_FINETUNE_PATIENCE),
        )
    except _EarlyStop:
        pass

    # Weight-only transfer to a fresh autoencoder for the target (GDEC's own recipe).
    autoencoder_transfer = StackedDenoisingAutoEncoder(dims, final_activation=None)
    src_dict = {k: v for k, v in autoencoder.state_dict().items() if "weight" in k}
    tar_dict = autoencoder_transfer.state_dict()
    tar_dict.update(src_dict)
    autoencoder_transfer.load_state_dict(tar_dict)
    if cuda:
        autoencoder_transfer.cuda()

    ae_optimizer_t = SGD(autoencoder_transfer.parameters(), lr=0.5, momentum=0.9)
    try:
        ae.train(
            ds_trg, autoencoder_transfer, cuda=cuda, validation=ds_trg, epochs=GDEC_TRANSFER_EPOCHS,
            batch_size=GDEC_BATCH_SIZE, optimizer=ae_optimizer_t,
            scheduler=StepLR(ae_optimizer_t, 30, gamma=0.1), corruption=0.2, silent=True,
            update_callback=_ValLossEarlyStopper(GDEC_TRANSFER_PATIENCE),
        )
    except _EarlyStop:
        pass

    transfer_model = DEC(cluster_number=K, hidden_dimension=GDEC_HIDDEN_DIMS[-1],
                          encoder=autoencoder_transfer.encoder)
    if cuda:
        transfer_model.cuda()
    dec_optimizer = SGD(transfer_model.parameters(), lr=0.001, momentum=0.9)
    dec_train(
        dataset=ds_trg, model=transfer_model, epochs=GDEC_DEC_EPOCHS, batch_size=GDEC_BATCH_SIZE,
        optimizer=dec_optimizer, cuda=cuda, silent=True, stopping_delta=GDEC_DEC_STOPPING_DELTA,
    )
    predicted, _actual = dec_predict(ds_trg, transfer_model, 1024, silent=True, return_actual=True,
                                      cuda=cuda)
    return predicted.cpu().numpy()


def method_gdec_gcnfree(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    X_S_pooled = np.vstack(sources)
    return _gdec_gcnfree_impl(X_T, X_S_pooled, seed)
