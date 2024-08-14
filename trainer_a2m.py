# -*- coding: utf-8 -*-
import os

import librosa
import numpy as np
import torch
import torchaudio
from torch import nn
from tqdm import tqdm
from Speecher.losses import calculate_loss_a2m, calculate_d_loss_a2m
from collections import defaultdict
import soundfile as sf


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self,
                 args,
                 model=None,
                 model_ema = None,
                 optimizer=None,
                 discriminator=None,
                 discriminator_optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False,
                 save_samples=False,
                 gen_dataloader=None,
                 emo_dataloader = None
                 ):
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        self.save_samples = save_samples
        self.gen_dataloader = gen_dataloader
        self.emo_dataloader = emo_dataloader

    def _train_epoch(self, encodec_model):
        """Train model one epoch."""
        raise NotImplementedError

    @torch.no_grad()
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        pass

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": self.model.state_dict()
        }
        if self.model_ema is not None:
            state_dict['model_ema'] = self.model_ema.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

        if self.model_ema is not None:
            self._load(state_dict["model_ema"], self.model_ema)

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm


    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def _train_epoch(self):
        self.epochs += 1

        train_losses = defaultdict(list)
        self.model.train()
        scaler = torch.cuda.amp.GradScaler() if (('cuda' in str(self.device)) and self.fp16_run) else None

        emo_data_iter = iter(self.emo_dataloader)

        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            """try:
                data = next(emo_data_iter)
            except StopIteration:
                emo_data_iter = iter(self.emo_dataloader)
                data = next(emo_data_iter)
            ### load data
            batch = [torch.cat((b, e), 0) for b, e in zip(batch, data)]"""
            batch = [b.to(self.device) for b in batch]
            x_real, label, spk_emb = batch
            x_real.requires_grad_()
            spk_emb.requires_grad_()
            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    d_loss, d_loss_latent = calculate_d_loss_a2m(self.model, self.discriminator, self.args, x_real, spk_emb, label)
                scaler.scale(d_loss).backward()
            else:
                d_loss, d_loss_latent = calculate_d_loss_a2m(self.model, self.discriminator, self.args, x_real, spk_emb, label)
                d_loss.backward()

            if scaler is not None:
                scaler.step(self.discriminator_optimizer)
                scaler.update()
            else:
                self.discriminator_optimizer.step()

            for key in d_loss_latent:
                train_losses["train_dis/%s" % key].append(d_loss_latent[key])


            if scaler is not None:
                with torch.autograd.set_detect_anomaly(True):
                    with torch.cuda.amp.autocast():
                        loss, loss_latent = calculate_loss_a2m(self.model, self.discriminator, self.args, x_real, spk_emb, label)
                    with torch.autograd.set_detect_anomaly(True):
                        scaler.scale(loss).backward()
            else:
                loss, loss_latent = calculate_loss_a2m(self.model,  self.discriminator, self.args, x_real, spk_emb, label)
                loss.backward()

            if scaler is not None:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.9)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.9)
                self.optimizer.step()

            if self.model_ema is not None:
                self.moving_average(self.model.f0_conv, self.model_ema.f0_conv, beta=0.999)
                self.moving_average(self.model.content_upsample, self.model_ema.content_upsample, beta=0.999)
                self.moving_average(self.model.decode, self.model_ema.decode, beta=0.999)
                self.moving_average(self.model.spk_emb_net, self.model_ema.spk_emb_net, beta=0.999)
                self.moving_average(self.model.spk_emb_net_unshared, self.model_ema.spk_emb_net_unshared, beta=0.999)
                self.moving_average(self.model.cont_emb_net, self.model_ema.cont_emb_net, beta=0.999)
                self.moving_average(self.model.to_out, self.model_ema.to_out, beta=0.999)


            for key in loss_latent:
                train_losses["train/%s" % key].append(loss_latent[key])

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        eval_losses = defaultdict(list)
        self.model.eval()
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            batch = [b.to(self.device) for b in batch]
            x_real, label, spk_emb = batch
            d_loss, d_loss_latent = calculate_d_loss_a2m(self.model, self.discriminator, self.args, x_real, spk_emb, label)
            for key in d_loss_latent:
                eval_losses["eval_dis/%s" % key].append(d_loss_latent[key])

            loss, loss_latent = calculate_loss_a2m(self.model, self.discriminator, self.args, x_real, spk_emb, label)

            for key in loss_latent:
                eval_losses["eval/%s" % key].append(loss_latent[key])

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        return eval_losses


    @torch.no_grad()
    def _sample_write_epoch(self, sample_write_params, vocoder, device='cpu',):
        if self.save_samples and self.gen_dataloader is not None and self.epochs % 2 == 0:
            self.model.eval()
            if "Emotional Speech Dataset (ESD)" in sample_write_params['real_sample_path']:
                base_str = "_Neutral_0012_000014_target_"
            elif "EmoDB" in sample_write_params['real_sample_path']:
                base_str = "_Angry_13_b02_target_"
            else:
                base_str = "_p258_101_target_p"
            for gen_steps_per_epoch, batch in enumerate(tqdm(self.gen_dataloader, desc="[generate sample]"), 1):
                ### load data
                batch = [b.to(self.device) for b in batch]
                _, y_trg, spk_emb = batch
                source_path = sample_write_params['real_sample_path']
                save_path = sample_write_params['sample_save_path']
                os.makedirs(save_path, exist_ok=True)
                audio, source_sr = librosa.load(source_path, sr=24000)
                if source_sr != 24000:
                    audio = librosa.resample(audio, source_sr, 24000)
                #audio = audio / np.max(np.abs(audio))
                audio.dtype = np.float32
                batch_size = spk_emb.size(0)
                real = self.preprocess(audio).to(device).unsqueeze(1).repeat(batch_size, 1, 1, 1)
                x_fake = self.model_ema(real, spk_emb, y_trg)
                x_fake = x_fake.transpose(-1, -2).squeeze()
                target_list = y_trg.cpu().numpy().tolist()
                speaker_target_map = {}
                for speakers in list(sample_write_params['selected_speakers']):
                    p, t = speakers.split("_")
                    speaker_target_map[int(t)] = p
                for idx, target in enumerate(target_list):
                    y_out = vocoder.inference(x_fake[idx].squeeze())
                    y_out = y_out.view(-1).cpu()
                    sf.write(
                        save_path + 'epoch_' + str(self.epochs) + "_" + str(idx) + "_" + base_str + speaker_target_map[
                            int(target)] + '.wav', y_out.numpy().squeeze(), 24000, 'PCM_24')
        return None

    @torch.no_grad()
    def preprocess(self, wave):
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor