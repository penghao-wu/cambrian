import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.modeling_llama import LlamaRMSNorm
class VisionMLP(nn.Module):
	def __init__(self, config, intermediate_size):
		super().__init__()
		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.proj = nn.Sequential(
			nn.Linear(intermediate_size*2, intermediate_size, bias=False),
			nn.SiLU(),
			nn.Linear(intermediate_size, config.hidden_size, bias=False)
		)
		self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(self, image_full, image_concise, side_len_full, side_len_concise, attention_mask=None):
		num_image_crops = image_full.shape[1]//side_len_full**2
		bs = image_full.shape[0]
		reduce_factor = side_len_full//side_len_concise

		image_full = image_full.view(bs*num_image_crops, side_len_full, side_len_full, -1)
		image_concise = image_concise.view(bs*num_image_crops, side_len_concise, side_len_concise, -1)
		image_concise = image_concise.repeat_interleave(reduce_factor, 1).repeat_interleave(reduce_factor, 2)
		image_concise = self.context_proj(image_concise)
		residual = image_full
		image_full = self.input_proj(image_full)
		image_full = torch.cat([image_full, image_concise], -1)
		image_full = self.layernorm_post(self.proj(image_full) + residual) 

		image_full = image_full.view(bs, num_image_crops*side_len_full*side_len_full, -1)

		return image_full
