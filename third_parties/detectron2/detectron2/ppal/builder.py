from detectron2.utils.registry import Registry
from detectron2.utils.logger import _log_api_usage

SAMPLER_REGISTRY = Registry("SAMPLER")

def build_al_sampler(cfg):
    
    sampler_ = cfg.NAME
    params = {k: v for k, v in cfg.items() if k != 'NAME'}      # 提取键值
    sampler = SAMPLER_REGISTRY.get(sampler_)(**params)       # ** 将字典转换为关键字参数
    _log_api_usage("modeling.al_sampler." + sampler_)
    
    return sampler