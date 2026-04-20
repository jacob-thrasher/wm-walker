import io
import config
import data_loader
import doy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from typing import Union
from models import IDM, Policy, WorldModel
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader


def obs_to_img(obs: Tensor) -> Tensor:
    return ((obs.permute(1, 2, 0) + 0.5) * 255).to(torch.uint8).numpy(force=True)


def create_decoder(in_dim, out_dim, device=config.DEVICE, hidden_sizes=(128, 128), state_dict: Union[dict, None] = None,):
    decoder = []
    in_size = h = in_dim
    for h in hidden_sizes:
        decoder.extend([nn.Linear(in_size, h), nn.ReLU()])
        in_size = h
    decoder.append(nn.Linear(h, out_dim))
    layers =  nn.Sequential(*decoder).to(device)

    if state_dict is not None:
        layers.load_state_dict(state_dict, strict=True)

    return layers

def create_dynamics_models(
    model_cfg: config.ModelConfig, state_dicts: Union[dict, None] = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (2 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_policy(
    model_cfg: config.ModelConfig,
    action_dim: int,
    policy_in_depth: int = 3,
    state_dict: Union[dict, None] = None,
    strict_loading: bool = True,
):
    policy = Policy(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)

    if state_dict is not None:
        policy.load_state_dict(state_dict, strict=strict_loading)

    return policy


def eval_latent_repr(labeled_data: data_loader.DataStager, idm: IDM):
    batch = labeled_data.td_unfolded[:131072]
    actions = idm.label_chunked(batch).select("ta", "la").to(config.DEVICE)
    return train_decoder(data=actions)


def train_decoder(
    data: TensorDict,  # tensordict with keys "la", "ta"
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    # TA_DIM = 15
    TA_DIM=6
    decoder = create_decoder(data["la"].shape[-1], TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)

    train_data, test_data = data[: len(data) // 2], data[len(data) // 2 :]

    dataloader = DataLoader(
        train_data,  # type: ignore
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    step = 0
    for i in range(epochs):
        for batch in dataloader:
            pred_ta = decoder(batch["la"])
            ta = batch["ta"][:, -2]
            # loss = F.cross_entropy(pred_ta, ta)
            loss = F.mse_loss(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                # train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            if step % 10 == 0:
                with torch.no_grad():
                    test_pred_ta = decoder(test_data["la"])
                    test_ta = test_data["ta"][:, -2]

                    logger(
                        step=i,
                        test_loss=F.mse_loss(test_pred_ta, test_ta), # F.cross_entropy(test_pred_ta, test_ta),
                        # test_acc=(test_pred_ta.argmax(-1) == test_ta).float().mean(),
                    )
            step += 1

    metrics = dict(
        # train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        # test_acc=logger["test_acc"][-1],
        test_loss=logger["test_loss"][-1],
    )

    return decoder, metrics


class VideoBuffer:
    def __init__(self):
        self.buffer = []
    
    def add_to_buffer(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = imageio.imread(buf)
        self.buffer.append(image)
        buf.close()

    def save_as_gif(self, filename='out.gif', duration=0.5):
        # The first image is the wrong size and I can't be bothered to figure out why
        imageio.mimsave(filename, self.buffer[1:], duration=duration)

    def clear_buffer(self):
        self.buffer.clear()