from __future__ import annotations

import torch


class AutoCLIPProcessor:
    """Small adapter over Hugging Face CLIPProcessor.

    The original project used a local `scripts.infer_clips` wrapper. This adapter
    covers standard CLIP-style checkpoints and keeps the same minimal interface.
    For OpenCLIP/timm/EVA custom backends, provide your own `scripts.infer_clips`
    earlier on `PYTHONPATH`.
    """

    def __init__(self, processor):
        self.processor = processor

    @classmethod
    def from_pretrained(cls, model_id: str):
        from transformers import CLIPProcessor

        return cls(CLIPProcessor.from_pretrained(model_id))

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)


class AutoCLIPModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, model_id: str):
        from transformers import CLIPModel

        return cls(CLIPModel.from_pretrained(model_id))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def load_finetuned_state_dict(self, state_dict, strict: bool = False):
        return self.model.load_state_dict(state_dict, strict=strict)

