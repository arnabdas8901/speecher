import sys
sys.path.append("/home/adas/Projects/StarGAN_v2/")
import yaml
from munch import Munch
from Speecher.model import Speecher_A2M

def count_parameters(model):
    params = list(model.f0_conv.parameters()) + list(model.content_upsample.parameters()) \
             + list(model.decode.parameters()) + list(model.to_out.parameters()) + list(model.spk_emb_net.parameters()) \
             + list(model.cont_emb_net.parameters()) + list(model.spk_emb_net_unshared.parameters())
    return sum(p.numel() for p in params if p.requires_grad)

config_path = "/home/adas/Projects/StarGAN_v2/Speecher/Config/config.yml"
config = yaml.safe_load(open(config_path))
model = Speecher_A2M(args=Munch(config['model_params']), asr_model=None, f0_model=None)

print("param count", count_parameters(model))