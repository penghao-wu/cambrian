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

	def forward(self, image_full, image_compress, compress_reduce_factor, per_crop_token_len=576, attention_mask=None):
		side_len_full = int(per_crop_token_len**0.5)
		side_len_compress = side_len_full // compress_reduce_factor

		num_image_crops = image_full.shape[1]//per_crop_token_len
		bs = image_full.shape[0]

		image_full = image_full.view(bs*num_image_crops, side_len_full, side_len_full, -1)
		image_compress = image_compress.view(bs*num_image_crops, side_len_compress, side_len_compress, -1)
		image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2)
		image_compress = self.context_proj(image_compress)
		residual = image_full
		image_full = self.input_proj(image_full)
		image_full = torch.cat([image_full, image_compress], -1)
		image_full = self.layernorm_post(self.proj(image_full) + residual) 

		image_full = image_full.view(bs, num_image_crops*side_len_full*side_len_full, -1)

		return image_full
