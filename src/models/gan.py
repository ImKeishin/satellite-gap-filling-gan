from __future__ import annotations

from src.models.discriminator import Discriminator
from src.models.generator import GeneratorUNet


def build_models(in_channels: int = 1, 
                 out_channels: int = 1, 
                 base_channels: int = 64):
    
    generator = GeneratorUNet(in_channels=in_channels,
                              out_channels=out_channels,
                              base_channels=base_channels)
    
    discriminator = Discriminator(condition_channels=in_channels,
                                  target_channels=out_channels,
                                  base_channels=base_channels)
    
    return generator, discriminator
