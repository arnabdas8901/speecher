import torch
from munch import Munch
from transforms import build_transforms
import torch.nn.functional as F
#from torchmetrics.functional.image import multiscale_structural_similarity_index_measure, structural_similarity_index_measure


def calculate_loss(model, discriminator, args, x_real, spk_emb):
    args = Munch(args)

    x_fake = model(x_real, spk_emb)
    loss_reco = torch.mean(torch.abs(x_fake - x_real.detach())) + torch.mean(torch.abs(x_fake**2 - x_real.detach()**2))

    # adversarial loss
    out = discriminator(x_fake)
    loss_adv = adv_loss(out, 1)

    t = build_transforms()
    x_real_t = t(x_real)
    x_fake_t = model(x_real_t, spk_emb)
    #loss_reco += torch.mean(torch.abs(x_fake_t - x_real_t.detach()))

    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm - x_real_norm.detach()) - args.norm_bias)) ** 2).mean()

    with torch.no_grad():
        data_range = max(x_fake.max()-x_fake.min(), x_real.max()-x_real.min())
    #ssim_loss = structural_similarity_index_measure(x_fake, x_real.detach(), kernel_size=[5, 7],
    #                                                           data_range=data_range.detach())
    # ssim doesn't work
    ssim_loss = torch.ones(1).mean()

    with torch.no_grad():
        F0_real, _, _ = model.f0_model(x_real)
    F0_fake, _, _ = model.f0_model(x_fake)
    loss_f0 = f0_loss(F0_fake, F0_real.detach())

    with torch.no_grad():
        ASR_real = model.asr_model.get_feature(x_real)
    ASR_fake = model.asr_model.get_feature(x_fake)
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real.detach())

    loss = args.lambda_reco * loss_reco + args.lambda_norm * loss_norm \
           + args.lambda_adv * loss_adv + args.lambda_mss * (1-ssim_loss) \
           + args.lambda_f0 * loss_f0 + args.lambda_asr * loss_asr

    return loss, Munch(reco=loss_reco.item(),
                       norm=loss_norm.item(),
                       adv= loss_adv.item(),
                       ssim = ssim_loss.item(),
                       f0 = loss_f0.item(),
                       asr = loss_asr.item())

def calculate_loss_a2m(model, discriminator, args, x_real, spk_emb, label):
    args = Munch(args)

    x_fake = model(x_real, spk_emb, label)
    loss_reco = torch.mean(torch.abs(x_fake - x_real.detach())) + torch.mean(torch.abs(x_fake**2 - x_real.detach()**2))

    # adversarial loss
    out = discriminator(x_fake, label)
    loss_adv = adv_loss(out, 1)

    t = build_transforms()
    x_real_t = t(x_real)
    x_fake_t = model(x_real_t, spk_emb, label)
    #loss_reco += torch.mean(torch.abs(x_fake_t - x_real_t.detach()))

    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm - x_real_norm.detach()) - args.norm_bias)) ** 2).mean()

    with torch.no_grad():
        data_range = max(x_fake.max()-x_fake.min(), x_real.max()-x_real.min())
    #ssim_loss = structural_similarity_index_measure(x_fake, x_real.detach(), kernel_size=[5, 7],
    #                                                           data_range=data_range.detach())
    # ssim doesn't work
    ssim_loss = torch.ones(1).mean()

    with torch.no_grad():
        F0_real, _, _ = model.f0_model(x_real)
    F0_fake, _, _ = model.f0_model(x_fake)
    loss_f0 = f0_loss(F0_fake, F0_real.detach())

    with torch.no_grad():
        ASR_real = model.asr_model.get_feature(x_real)
    ASR_fake = model.asr_model.get_feature(x_fake)
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real.detach())

    loss = args.lambda_reco * loss_reco + args.lambda_norm * loss_norm \
           + args.lambda_adv * loss_adv + args.lambda_mss * (1-ssim_loss) \
           + args.lambda_f0 * loss_f0 + args.lambda_asr * loss_asr

    return loss, Munch(reco=loss_reco.item(),
                       norm=loss_norm.item(),
                       adv= loss_adv.item(),
                       ssim = ssim_loss.item(),
                       f0 = loss_f0.item(),
                       asr = loss_asr.item())

def calculate_d_loss(model, discriminator, args, x_real, spk_emb):
    x_real.requires_grad_()
    with torch.no_grad():
        x_fake = model(x_real, spk_emb)
    out = discriminator(x_real)
    loss_real = adv_loss(out, 1)
    out = discriminator(x_fake)
    loss_fake = adv_loss(out, 0)

    t = build_transforms()
    out_aug = discriminator(t(x_real).detach())
    loss_con_reg = F.smooth_l1_loss(out, out_aug)

    loss = loss_real + loss_fake + args.lambda_con_reg*loss_con_reg

    return loss, Munch(loss_real=loss_real.item(),
                       loss_fake=loss_fake.item(),
                       loss_con_reg = loss_con_reg.item())

def calculate_d_loss_a2m(model, discriminator, args, x_real, spk_emb, label):
    x_real.requires_grad_()
    with torch.no_grad():
        x_fake = model(x_real, spk_emb, label)
    out = discriminator(x_real, label)
    loss_real = adv_loss(out, 1)
    out = discriminator(x_fake, label)
    loss_fake = adv_loss(out, 0)

    t = build_transforms()
    out_aug = discriminator(t(x_real).detach(), label)
    loss_con_reg = F.smooth_l1_loss(out, out_aug)

    loss = loss_real + loss_fake + args.lambda_con_reg*loss_con_reg

    return loss, Munch(loss_real=loss_real.item(),
                       loss_fake=loss_fake.item(),
                       loss_con_reg = loss_con_reg.item())

def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-5, max=5) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss

def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

