import copy
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import os.path as osp
import sys
sys.path.append("/home/adas/Projects/StarGAN_v2/")
import yaml
import shutil
import torch
import click
import warnings
import torch.nn as nn
import numpy as np
warnings.simplefilter('ignore')

from munch import Munch

from Speecher.dataset import build_dataloader
from Speecher.model import Speecher_A2M, Discriminator2d, Discriminator2dA2M
from Speecher.trainer_a2m import Trainer
from torch.utils.tensorboard import SummaryWriter
from speechbrain.pretrained import EncoderClassifier

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from parallel_wavegan.utils import load_model

import logging
from logging import StreamHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('-p', '--config_path', default='/home/adas/Projects/StarGAN_v2/Speecher/Config/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    train_path = config.get('train_data', None)
    emo_train_path = config.get('emotional_train_data', None)
    val_path = config.get('val_data', None)
    fp16_run = config.get('fp16_run', False)
    sample_generation_param = config.get('sample_write_params', False)
    vocoder_path = config.get('vocoder_path', None)
    checkpoint_path = config.get("checkpoint_path", "")
    print("Configs loaded")

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    classifier.eval()
    print("Speaker encoder loaded")

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=32,
                                        device=device,
                                        spkr_model=classifier
                                        )
    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=32,
                                      device=device,
                                      spkr_model=classifier)
    _, selected_val_list = get_data_path_list(train_path, sample_generation_param['sample_generate_data'])
    gen_dataloader = build_dataloader(selected_val_list,
                                      batch_size=7,
                                      num_workers=1,
                                      device=device,
                                      spkr_model=classifier)
    emo_train_list, _ = get_data_path_list(emo_train_path, val_path)
    emo_train_dataloader = build_dataloader(emo_train_list,
                                            batch_size=batch_size,
                                            num_workers=6,
                                            device=device,
                                            spkr_model=classifier)

    print("Data loaders instantiated")

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    with open(ASR_config) as f:
        ASR_config = yaml.safe_load(f)
    ASR_model_config = ASR_config['model_params']
    ASR_model = ASRCNN(**ASR_model_config)
    params = torch.load(ASR_path, map_location='cpu')['model']
    ASR_model.load_state_dict(params)
    ASR_model.to(device)
    print("ASR loaded")

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(F0_path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    F0_model.to(device)
    print("F0 model loaded")

    with open(config.get('vocoder_config')) as f:
        voc_config = yaml.load(f, Loader=yaml.Loader)
    vocoder = load_model(vocoder_path, config=voc_config)
    vocoder.remove_weight_norm()
    vocoder.to(device)
    print("Vocoder loaded")

    model = Speecher_A2M(args=Munch(config['model_params']), asr_model=ASR_model, f0_model=F0_model)
    if len(checkpoint_path)>0:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        _load(state_dict["model"], model)
        print("Checkpoint loaded")
    model.to(device)
    lr = config['optimizer_params'].get('lr', 1e-3)
    discriminator_lr = config['optimizer_params'].get('discriminator_lr', lr)
    weight_decay = config['optimizer_params'].get('weight_decay', 1e-4)
    params = list(model.f0_conv.parameters()) + list(model.content_upsample.parameters()) \
             + list(model.decode.parameters()) + list(model.to_out.parameters()) + list(model.spk_emb_net.parameters()) \
             + list(model.cont_emb_net.parameters()) + list(model.spk_emb_net_unshared.parameters())
    discriminator = Discriminator2dA2M()
    discriminator.to(device)
    print("Model and discriminator instantiated")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-09)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-09)

    trainer = Trainer(args=Munch(config['loss_params']),
                                          model=model,
                                          model_ema=copy.deepcopy(model),
                                          optimizer=optimizer,
                                          discriminator = discriminator,
                                          discriminator_optimizer = discriminator_optimizer,
                                          device=device,
                                          train_dataloader=train_dataloader,
                                          val_dataloader=val_dataloader,
                                          logger=logger,
                                          fp16_run=fp16_run,
                                          save_samples=config['save_samples'],
                                          gen_dataloader=gen_dataloader,
                                          emo_dataloader=emo_train_dataloader)

    print("Training Starts")
    for _ in range(1, epochs+1):
        epoch = trainer.epochs
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        _ = trainer._sample_write_epoch(config['sample_write_params'], vocoder, device= device)
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        txt = ''
        for key, value in results.items():
            if isinstance(value, float):
                txt = txt + key + ':'+ format(value, ".4f") + '  '
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_spec', v, epoch)
        logger.info(txt)
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))

    return 0






def get_data_path_list(train_path, val_path):
    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list

def _load(states, model, force_load=True):
    model_states = model.state_dict()
    for key, val in states.items():
        try:
            if key not in model_states:
                continue
            if isinstance(val, nn.Parameter):
                val = val.data

            if val.shape != model_states[key].shape:
                print(val.shape, model_states[key].shape)
                if not force_load:
                    continue

                min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                slices = [slice(0, min_index) for min_index in min_shape]
                model_states[key][slices].copy_(val[slices])
            else:
                model_states[key].copy_(val)
        except:
            print("not exist ", key)

if __name__=="__main__":

    main()