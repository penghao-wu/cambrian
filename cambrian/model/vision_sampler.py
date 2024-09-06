import torch
import torch.utils.checkpoint
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
	"""
	grid_size: int of the grid height and width
	return:
	pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
	"""
	grid_h = np.arange(grid_size, dtype=np.float32)
	grid_w = np.arange(grid_size, dtype=np.float32)
	grid = np.meshgrid(grid_w, grid_h)  # here w goes first
	grid = np.stack(grid, axis=0)

	grid = grid.reshape([2, 1, grid_size, grid_size])

	pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
	if cls_token:
		pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
	return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
	assert embed_dim % 2 == 0

	# use half of dimensions to encode grid_h
	emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
	emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

	emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
	return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
	"""
	embed_dim: output dimension for each position
	pos: a list of positions to be encoded: size (M,)
	out: (M, D)
	"""
	assert embed_dim % 2 == 0
	omega = np.arange(embed_dim // 2, dtype=np.float32)
	omega /= embed_dim / 2.
	omega = 1. / 10000 ** omega  # (D/2,)

	pos = pos.reshape(-1)  # (M,)
	out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

	emb_sin = np.sin(out)  # (M, D/2)
	emb_cos = np.cos(out)  # (M, D/2)

	emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
	return emb


class CrossAttention(nn.Module):

	def __init__(self, q_dim, kv_dim, hidden_dim, num_heads, out_dim=None, attention_bias=False):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.head_dim = self.hidden_dim // self.num_heads

		if out_dim is None:
			out_dim = q_dim

		if (self.head_dim * self.num_heads) != self.hidden_dim:
			raise ValueError(
				f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
				f" and `num_heads`: {self.num_heads})."
			)

		self.q_proj = nn.Sequential(nn.Linear(q_dim, self.num_heads * self.head_dim, bias=attention_bias))
		self.k_proj = nn.Sequential(nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
		self.v_proj = nn.Sequential(nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
		self.o_proj = nn.Linear(self.num_heads * self.head_dim, out_dim, bias=attention_bias)

	def forward(
		self,
		vision_latents, queries, attention_mask=None
	):
		
		bsz, q_len, _ = queries.size()
		bsz, v_len, _ = vision_latents.size()

		query_states = self.q_proj(queries)
		key_states = self.k_proj(vision_latents)
		value_states = self.v_proj(vision_latents)

		query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = key_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)


		if attention_mask is not None:
			if attention_mask.size() != (bsz, 1, q_len, v_len):
				raise ValueError(
					f"Attention mask should be of size {(bsz, 1, q_len, v_len)}, but is {attention_mask.size()}"
				)

		# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
		# Reference: https://github.com/pytorch/pytorch/issues/112577.
		if query_states.device.type == "cuda" and attention_mask is not None:
			query_states = query_states.contiguous()
			key_states = key_states.contiguous()
			value_states = value_states.contiguous()

		attn_output = torch.nn.functional.scaled_dot_product_attention(
			query_states,
			key_states,
			value_states,
			attn_mask=attention_mask,
		)

		attn_output = attn_output.transpose(1, 2).contiguous()
		attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

		attn_output = self.o_proj(attn_output)
		return attn_output
	

class AggregationBlock(nn.Module):
	def __init__(self, attention, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.head_dim = self.hidden_dim // self.num_heads

		if (self.head_dim * self.num_heads) != self.hidden_dim:
			raise ValueError(
				f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
				f" and `num_heads`: {self.num_heads})."
			)

		self.attention = attention
		if attention:
			self.attention_layer = CrossAttention(q_dim, kv_dim, hidden_dim, num_heads, attention_bias)
		else:
			self.attention_layer = MLP(kv_dim, q_dim, q_dim)        

	def forward(
		self,
		vision_latents, queries, attention_mask
	):
		if self.attention:
			queries = self.attention_layer(vision_latents, queries, attention_mask)
		else:
			queries = self.attention_layer(vision_latents)

		return queries


class MultiKVCrossAttention(nn.Module):

	def __init__(self, q_dim, kv_dim_list, hidden_dim, num_heads, attention_bias=False):
		super().__init__()

		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.head_dim = self.hidden_dim // self.num_heads

		if (self.head_dim * self.num_heads) != self.hidden_dim:
			raise ValueError(
				f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
				f" and `num_heads`: {self.num_heads})."
			)

		self.q_proj = nn.Sequential(nn.LayerNorm(q_dim), nn.Linear(q_dim, self.num_heads * self.head_dim, bias=attention_bias))
		self.num_of_kvs = len(kv_dim_list)
		for i, kv_dim in enumerate(kv_dim_list):
			setattr(self, 'k_proj_{}'.format(i), nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias)))
			setattr(self, 'v_proj_{}'.format(i), nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias)))
		self.o_proj = nn.Linear(self.num_heads * self.head_dim, q_dim, bias=attention_bias)

	def forward(
		self,
		queries, *vision_latents_attention_mask_list,
	):
		
		vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
		attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]
		
		bsz, q_len, _ = queries.size()

		query_states = self.q_proj(queries)
		key_states = torch.cat([getattr(self, 'k_proj_{}'.format(i))(vision_latents_list[i]) for i in range(self.num_of_kvs)], dim=1)
		value_states = torch.cat([getattr(self, 'v_proj_{}'.format(i))(vision_latents_list[i]) for i in range(self.num_of_kvs)], dim=1)

		v_len = key_states.shape[1]

		query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = key_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)

		# if kv_weight is not None:
		#     kv_weight = kv_weight.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

		attention_mask = torch.cat(attention_mask_list, dim=-1)

		if attention_mask is not None:
			if attention_mask.size() != (bsz, 1, q_len, v_len):
				raise ValueError(
					f"Attention mask should be of size {(bsz, 1, q_len, v_len)}, but is {attention_mask.size()}"
				)

		# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
		# Reference: https://github.com/pytorch/pytorch/issues/112577.
		if query_states.device.type == "cuda" and attention_mask is not None:
			query_states = query_states.contiguous()
			key_states = key_states.contiguous()
			value_states = value_states.contiguous()

		attn_output = torch.nn.functional.scaled_dot_product_attention(
			query_states,
			key_states,
			value_states,
			attn_mask=attention_mask,
		)
		# attn_output = spda(
		#     query_states,
		#     key_states,
		#     value_states,
		#     attn_mask=attention_mask,
		#     additional_score=kv_weight
		# )

		attn_output = attn_output.transpose(1, 2).contiguous()
		attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

		attn_output = self.o_proj(attn_output)

		return attn_output


class MLP(nn.Module):
	def __init__(self, d_in, d_hidden, d_out):
		super().__init__() 
		self.linear_1 = nn.Linear(d_in, d_hidden, bias=False)
		self.act = nn.GELU()
		self.linear_2 = nn.Linear(d_hidden, d_out, bias=False)

	def forward(self, x):
		return self.linear_2(self.act(self.linear_1(x)))


class VisionCrossAttentionLayer(nn.Module):
	def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, hidden_dim = 1024, layer_idx=0, gist_token=False):
		super().__init__()
		num_heads = 16
		self.num_of_kvs = len(kv_dim_list)

		self.proj_context = nn.Linear(context_dim, hidden_dim, bias=False)
		if gist_token:
			self.proj_gist = nn.Linear(q_dim, hidden_dim, bias=False)
			self.proj_in = nn.Linear(q_dim+hidden_dim*2, hidden_dim, bias=False)
		else:
			self.proj_in = nn.Linear(q_dim+hidden_dim, hidden_dim, bias=False)
		# if self.num_of_kvs > 1:
		#     self.weight_mlp = MLP(q_dim+hidden_dim, hidden_dim, self.num_of_kvs)
		#     self.tower_weight = nn.Parameter(torch.zeros((self.num_of_kvs)))
		self.proj_out = MLP(hidden_dim, hidden_dim, q_dim)

		self.norm = nn.LayerNorm(hidden_dim)

		self.cross_attn = MultiKVCrossAttention(hidden_dim, kv_dim_list, hidden_dim, num_heads)
		self.kv_size_list = kv_size_list
		for i, kv_size in enumerate(kv_size_list):
			if kv_size > 1:
				# setattr(self, "pos_embed_{}".format(i), nn.Parameter(torch.randn(kv_size**2, hidden_dim)))
				self.register_buffer("pos_embed_{}".format(i), torch.from_numpy(get_2d_sincos_pos_embed(hidden_dim, kv_size)).float(), persistent=False)

	def forward(
		self,
		queries,
		context_feature,
		gist_token=None,
		*vision_latents_attention_mask_list,
	) -> torch.FloatTensor:
		residual = queries
		# queries = self.proj_in(queries)
		context_feature = self.proj_context(context_feature)
		if gist_token is not None:
			gist_token = self.proj_gist(gist_token)
			queries = torch.cat([queries, context_feature, gist_token], -1)
		else:
			# queries = queries + context_feature
			queries = torch.cat([queries, context_feature], -1)

		# if self.num_of_kvs > 1:
		#     kv_weight = self.weight_mlp(queries) # B * 1 * num_tower
		#     kv_weight = kv_weight + self.tower_weight.view(1, 1, -1)
		#     kv_weight = kv_weight.softmax(-1)
		#     kv_number_list = [size**2 for size in self.kv_size_list]
		#     kv_weight = torch.repeat_interleave(kv_weight, torch.tensor(kv_number_list).to(kv_weight.device), dim=-1)
		# else:
		#     kv_weight = None

		queries = self.proj_in(queries)

		vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
		attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]

		attention_mask_list_reshaped = []
		if attention_mask_list is not None:
			for attention_mask in attention_mask_list:
				attention_mask = attention_mask.view(attention_mask.shape[0], 1, 1, -1)
				attention_mask = attention_mask.expand(-1, -1, queries.shape[1], -1)
				attention_mask_list_reshaped.append(attention_mask)

		vision_latents_pos_list = []
		for i, vision_latents in enumerate(vision_latents_list):
			if vision_latents.shape[1] > 1:
				vision_latents_pos_list.append(vision_latents + getattr(self, "pos_embed_{}".format(i))[None, :, :].to(vision_latents.dtype))
			else:
				vision_latents_pos_list.append(vision_latents)

		# Cross Attention
		attention_output = self.cross_attn(
			queries,
			*vision_latents_pos_list,
			*attention_mask_list_reshaped
		)

		# attention_output = (attention_output * combination_weight).sum(2)
		queries = queries + attention_output

		queries = self.norm(queries)

		queries = self.proj_out(queries)

		queries = queries + residual

		return queries


class VisionAggregationLayer(nn.Module):
	def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, hidden_dim = 1024, layer_idx=0):
		super().__init__()
		num_heads = 16
		self.num_of_kvs = len(kv_dim_list)

		self.proj_context = nn.Linear(context_dim, hidden_dim, bias=False)
		self.proj_in = nn.Linear(q_dim+hidden_dim, hidden_dim, bias=False)

		self.proj_out = MLP(hidden_dim, hidden_dim, q_dim)

		self.norm = nn.LayerNorm(hidden_dim)

		if self.num_of_kvs > 1:
			self.weight_mlp = MLP(q_dim+hidden_dim, hidden_dim, self.num_of_kvs)

		for i, kv_size in enumerate(kv_size_list):
			if kv_size > 1:
				setattr(self, "pos_embed_{}".format(i), nn.Parameter(torch.randn(kv_size**2, hidden_dim)))
				setattr(self, "aggregate_{}".format(i), AggregationBlock(True, hidden_dim, kv_dim_list[i], hidden_dim, num_heads))
			else:
				setattr(self, "aggregate_{}".format(i), AggregationBlock(False, hidden_dim, kv_dim_list[i], hidden_dim, num_heads))

	def forward(
		self,
		queries,
		context_feature,
		*vision_latents_attention_mask_list,
	) -> torch.FloatTensor:

		residual = queries
		# queries = self.proj_in(queries)
		context_feature = self.proj_context(context_feature)
		# queries = queries + context_feature
		queries = torch.cat([queries, context_feature], -1)

		if self.num_of_kvs > 1:
			combination_weight = self.weight_mlp(queries).softmax(-1) # B * 1 * num_tower
			combination_weight = combination_weight.unsqueeze(-1)
		else:
			combination_weight = 1

		queries = self.proj_in(queries)

		vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
		attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]

		attention_mask_list_reshaped = []
		if attention_mask_list is not None:
			for attention_mask in attention_mask_list:
				attention_mask = attention_mask.view(attention_mask.shape[0], 1, 1, -1)
				attention_mask = attention_mask.expand(-1, -1, queries.shape[1], -1)
				attention_mask_list_reshaped.append(attention_mask)

		vision_latents_pos_list = []
		for i, vision_latents in enumerate(vision_latents_list):
			if vision_latents.shape[1] > 1:
				vision_latents_pos_list.append(vision_latents + getattr(self, "pos_embed_{}".format(i))[None, :, :].to(vision_latents.dtype))
			else:
				vision_latents_pos_list.append(vision_latents)

		aggregated_vision_latents_list = []
		for i, (vision_latents, attention_mask) in enumerate(zip(vision_latents_pos_list,attention_mask_list_reshaped)):
			aggregated_vision_latents_list.append(getattr(self, "aggregate_{}".format(i))(vision_latents, queries, attention_mask))

		aggregated_vision_latents = torch.stack(aggregated_vision_latents_list, 2)

		queries = queries + (aggregated_vision_latents * combination_weight).sum(2)

		queries = self.norm(queries)

		queries = self.proj_out(queries)

		queries = queries + residual

		return queries

class VisionTokenSampler(nn.Module):
	def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, num_of_layers=1, gist_token=False, layer_type="joint"):
		super().__init__()
		assert layer_type in ['joint', 'sep']
		if layer_type == 'joint':
			self.layers = nn.ModuleList([VisionCrossAttentionLayer(q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, idx, gist_token) for idx in range(num_of_layers)])
		else:
			self.layers = nn.ModuleList([VisionAggregationLayer(q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, idx) for idx in range(num_of_layers)])

	def forward(self, queries, context_feature, *vision_latents_attention_mask_list):
		for layer in self.layers:
			queries = layer(queries, context_feature, *vision_latents_attention_mask_list)
		return queries



from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaDecoderLayer, LlamaRMSNorm, rotate_half, repeat_kv
class VisionMLP(nn.Module):
	def __init__(self, config, intermediate_size=1024):
		super().__init__()
		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.proj = nn.Sequential(
			nn.Linear(intermediate_size*2, intermediate_size, bias=False),
			nn.SiLU(),
			nn.Linear(intermediate_size, config.hidden_size, bias=False)
		)
		# self.layernorm_pre = LlamaRMSNorm(intermediate_size*2, eps=config.rms_norm_eps)
		self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

		self.layernorm_pre = nn.Identity()
		# self.layernorm_post  = nn.Identity()

	def forward(self, input_embed, context, side_len_input, side_len_context, attention_mask=None):
		bs = input_embed.shape[0]
		reduce_factor = side_len_input//side_len_context

		input_embed = input_embed.view(bs, side_len_input, side_len_input+1, -1)
		context = context.view(bs, side_len_context, side_len_context+1, -1)

		input_embed = input_embed[:, :, :-1].view(bs, side_len_input, side_len_input, -1)
		input_embed = input_embed.view(bs, side_len_context, reduce_factor, side_len_context, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().flatten(1, 4)

		context_newline = context[:, :, -1:]
		context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, reduce_factor, reduce_factor, 1).flatten(1, 4)

		context = self.context_proj(context)
		residual = input_embed
		input_embed = self.input_proj(input_embed)
		input_embed = self.layernorm_pre(torch.cat([input_embed, context], -1))
		# input_embed = self.layernorm_pre(input_embed)
		# input_embed = input_embed + context
		input_embed = self.layernorm_post(self.proj(input_embed) + residual) 
		
		input_embed = input_embed.view(bs, side_len_context, side_len_context, reduce_factor, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, side_len_input, side_len_input, -1)

		input_embed_newline = torch.repeat_interleave(context_newline, reduce_factor, 1)

		input_embed = torch.cat([input_embed, input_embed_newline], 2).flatten(1,2)

		return input_embed


class VisionMLP_sa(nn.Module):
	def __init__(self, config, intermediate_size=1024):
		super().__init__()
		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.proj = nn.Sequential(
			nn.Linear(intermediate_size*2, config.hidden_size, bias=False),
		)

	def forward(self, input_embed, context, side_len_input, side_len_context, attention_mask=None):
		bs = input_embed.shape[0]
		reduce_factor = side_len_input//side_len_context

		input_embed = input_embed.view(bs, side_len_input, side_len_input+1, -1)
		context = context.view(bs, side_len_context, side_len_context+1, -1)

		input_embed = input_embed[:, :, :-1].view(bs, side_len_input, side_len_input, -1)
		input_embed = input_embed.view(bs, side_len_context, reduce_factor, side_len_context, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().flatten(1, 4)

		context_newline = context[:, :, -1:]
		context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, reduce_factor, reduce_factor, 1).flatten(1, 4)

		context = self.context_proj(context)
		input_embed = self.input_proj(input_embed)
		input_embed = torch.cat([input_embed, context], -1)
		input_embed = self.proj(input_embed) 
		
		input_embed = input_embed.view(bs, side_len_context, side_len_context, reduce_factor, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, side_len_input, side_len_input, -1)

		input_embed_newline = torch.repeat_interleave(context_newline, reduce_factor, 1)

		input_embed = torch.cat([input_embed, input_embed_newline], 2).flatten(1,2)

		return input_embed
	

class VisionMLP_ffn(nn.Module):
	def __init__(self, config, intermediate_size=1024):
		super().__init__()
		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.proj = nn.Sequential(
			nn.Linear(intermediate_size*2, intermediate_size, bias=False),
			nn.SiLU(),
			nn.Linear(intermediate_size, config.hidden_size, bias=False)
		)

	def forward(self, input_embed, context, side_len_input, side_len_context, attention_mask=None):
		bs = input_embed.shape[0]
		reduce_factor = side_len_input//side_len_context

		input_embed = input_embed.view(bs, side_len_input, side_len_input+1, -1)
		context = context.view(bs, side_len_context, side_len_context+1, -1)

		input_embed = input_embed[:, :, :-1].view(bs, side_len_input, side_len_input, -1)
		input_embed = input_embed.view(bs, side_len_context, reduce_factor, side_len_context, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().flatten(1, 4)

		context_newline = context[:, :, -1:]
		context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, reduce_factor, reduce_factor, 1).flatten(1, 4)

		context = self.context_proj(context)
		input_embed = self.input_proj(input_embed)
		input_embed = torch.cat([input_embed, context], -1)
		input_embed = self.proj(input_embed) 
		
		input_embed = input_embed.view(bs, side_len_context, side_len_context, reduce_factor, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, side_len_input, side_len_input, -1)

		input_embed_newline = torch.repeat_interleave(context_newline, reduce_factor, 1)

		input_embed = torch.cat([input_embed, input_embed_newline], 2).flatten(1,2)

		return input_embed
	
# class VisionMLP(nn.Module):
# 	def __init__(self, config, intermediate_size=1024):
# 		super().__init__()
# 		# self.sa = VisionMLP_sa(config, intermediate_size)
# 		self.sa = VisionSA(config, intermediate_size)
# 		self.ffn = nn.Sequential(
# 			nn.Linear(config.hidden_size, intermediate_size, bias=False),
# 			nn.SiLU(),
# 			nn.Linear(intermediate_size, config.hidden_size, bias=False)
# 		)
# 		# self.ffn = VisionMLP_ffn(config, intermediate_size)

# class VisionSA(nn.Module):
# 	def __init__(self, config, intermediate_size=1024):
# 		super().__init__()
# 		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		self.proj = nn.Sequential(
# 			nn.Linear(intermediate_size, intermediate_size, bias=False),
# 			nn.SiLU(),
# 			nn.Linear(intermediate_size, config.hidden_size, bias=False)
# 		)
# 		self.self_attention = CrossAttention(intermediate_size, intermediate_size, intermediate_size, 16)
# 		self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# 	def forward(self, input_embed, context, side_len_input, side_len_context, attention_masks=None):
# 		bs = input_embed.shape[0]
# 		reduce_factor = side_len_input//side_len_context

# 		input_embed = input_embed.view(bs, side_len_input, side_len_input+1, -1)
# 		context = context.view(bs, side_len_context, side_len_context+1, -1)

# 		input_embed = input_embed[:, :, :-1].view(bs, side_len_input, side_len_input, -1)
# 		input_embed = input_embed.view(bs, side_len_context, reduce_factor, side_len_context, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().flatten(0, 2).flatten(1, 2)

# 		context_newline = context[:, :, -1:]
# 		context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, 1, 1, 1).flatten(0, 2).flatten(1, 2)

# 		context = self.context_proj(context)
# 		residual = input_embed
# 		input_embed = self.input_proj(input_embed)
		
# 		if attention_masks is not None:
# 			attention_masks = attention_masks.view(bs*side_len_context*side_len_context, 1, 1, -1)
# 			attention_masks = attention_masks.repeat(1, 1, reduce_factor*reduce_factor, 1)

# 		sa_kv = torch.cat([input_embed, context], dim=1)
# 		input_embed = input_embed + self.self_attention(sa_kv, input_embed, attention_masks)

# 		input_embed = self.proj(input_embed) + residual
		
# 		input_embed = input_embed.view(bs, side_len_context, side_len_context, reduce_factor, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, side_len_input, side_len_input, -1)

# 		input_embed = self.layernorm_post(input_embed)

# 		input_embed_newline = torch.repeat_interleave(context_newline, reduce_factor, 1)

# 		input_embed = torch.cat([input_embed, input_embed_newline], 2).flatten(1,2)

# 		return input_embed
	

class VisionSA(nn.Module):
	def __init__(self, config, intermediate_size=1024):
		super().__init__()
		# self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.context_proj = nn.Identity()
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		# self.cat_proj = nn.Linear(intermediate_size*2, intermediate_size, bias=False)
		self.gate = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size, bias=False), nn.Sigmoid())
		self.self_attention = CrossAttention(intermediate_size, intermediate_size, intermediate_size, 16, config.hidden_size)
		self.layernorm_pre = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		# self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(self, input_embed, context, side_len_input, side_len_context, attention_masks=None):
		bs = input_embed.shape[0]
		reduce_factor = side_len_input//side_len_context
		
		input_embed = self.layernorm_pre(input_embed)

		input_embed = input_embed.view(bs, side_len_input, side_len_input+1, -1)
		context = context.view(bs, side_len_context, side_len_context+1, -1)

		input_embed = input_embed[:, :, :-1].view(bs, side_len_input, side_len_input, -1)
		
		input_embed = input_embed.view(bs, side_len_context, reduce_factor, side_len_context, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().flatten(0, 2).flatten(1, 2)
		residual = input_embed

		context_newline = context[:, :, -1:]
		context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, 1, 1, 1).flatten(0, 2).flatten(1, 2)
		# context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, reduce_factor, reduce_factor, 1).flatten(0, 2).flatten(1, 2)

		context = self.context_proj(context)
		input_embed = self.input_proj(input_embed)

		# input_embed = torch.cat([context, input_embed], -1)
		# input_embed = self.cat_proj(input_embed)
		
		if attention_masks is not None:
			attention_masks = attention_masks.view(bs*side_len_context*side_len_context, 1, 1, -1)
			attention_masks = attention_masks.repeat(1, 1, reduce_factor*reduce_factor, 1)
		# attention_masks = attention_masks[:, :, :, :-1]

		# sa_kv = torch.cat([input_embed, context], dim=1)
		sa_kv = input_embed
		# input_embed = self.self_attention(sa_kv, input_embed, attention_masks) + residual
		input_embed = self.self_attention(sa_kv, input_embed, attention_masks)
		gate_weight = self.gate(input_embed+context)
		input_embed = gate_weight*input_embed + (1-gate_weight)*context
		input_embed = input_embed + residual

		input_embed = input_embed.view(bs, side_len_context, side_len_context, reduce_factor, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, side_len_input, side_len_input, -1)

		input_embed_newline = torch.repeat_interleave(context_newline, reduce_factor, 1)

		input_embed = torch.cat([input_embed, input_embed_newline], 2).flatten(1,2)

		return input_embed




# class VisionSA(nn.Module):
# 	def __init__(self, config, intermediate_size=1024):
# 		super().__init__()
# 		# self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		# self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		# self.cat_proj = nn.Linear(intermediate_size*2, intermediate_size, bias=False)
# 		self.self_attention = CrossAttention(config.hidden_size, config.hidden_size, intermediate_size, 16, config.hidden_size)
# 		self.gate = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size, bias=False), nn.Sigmoid())
# 		# self.layernorm_pre = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
# 		# self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# 	def forward(self, input_embed, context, side_len_input, side_len_context, attention_masks=None):
# 		bs = input_embed.shape[0]
# 		reduce_factor = side_len_input//side_len_context
		
# 		input_embed = input_embed

# 		input_embed = input_embed.view(bs, side_len_input, side_len_input+1, -1)
# 		context = context.view(bs, side_len_context, side_len_context+1, -1)

# 		input_embed = input_embed[:, :, :-1].view(bs, side_len_input, side_len_input, -1)
		
# 		input_embed = input_embed.view(bs, side_len_context, reduce_factor, side_len_context, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().flatten(0, 2).flatten(1, 2)
# 		residual = input_embed

# 		context_newline = context[:, :, -1:]
# 		# context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, 1, 1, 1).flatten(0, 2).flatten(1, 2)
# 		context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, reduce_factor, reduce_factor, 1).flatten(0, 2).flatten(1, 2)

# 		# context = self.context_proj(context)
# 		# input_embed = self.input_proj(input_embed)

# 		# input_embed = torch.cat([context, input_embed], -1)
# 		# input_embed = self.cat_proj(input_embed)
		
# 		if attention_masks is not None:
# 			attention_masks = attention_masks.view(bs*side_len_context*side_len_context, 1, 1, -1)
# 			attention_masks = attention_masks.repeat(1, 1, reduce_factor*reduce_factor, 1)
# 		attention_masks = attention_masks[:, :, :, :-1]

# 		# sa_kv = torch.cat([input_embed, context], dim=1)
# 		sa_kv = input_embed
# 		input_embed = self.self_attention(sa_kv, input_embed, attention_masks)
# 		gate_weight = self.gate(input_embed+context)
# 		input_embed = gate_weight*input_embed + (1-gate_weight)*context

# 		input_embed = input_embed.view(bs, side_len_context, side_len_context, reduce_factor, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, side_len_input, side_len_input, -1)

# 		input_embed_newline = torch.repeat_interleave(context_newline, reduce_factor, 1)

# 		input_embed = torch.cat([input_embed, input_embed_newline], 2).flatten(1,2)

# 		return input_embed

class CrossNorm(nn.Module):
    def __init__(self, C, epsilon=1e-5, affine=False):
        super(CrossNorm, self).__init__()
        self.epsilon = epsilon
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, C))
            # self.beta = nn.Parameter(torch.zeros(1, 1, C))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x, ref):
        mean = ref.mean(dim=1, keepdim=True)
        variance = ref.var(dim=1, unbiased=False, keepdim=True)
        
        x_normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        
        if self.affine:
            x_normalized = x_normalized * self.gamma
        
        return x_normalized

class VisionMLP_crossnorm(nn.Module):
	def __init__(self, config, intermediate_size=1024):
		super().__init__()
		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.proj = nn.Sequential(
			nn.Linear(intermediate_size*2, intermediate_size, bias=False),
			nn.SiLU(),
			nn.Linear(intermediate_size, config.hidden_size, bias=False)
		)
		self.layernorm_pre = LlamaRMSNorm(intermediate_size*2, eps=config.rms_norm_eps)
		self.cross_norm_post = CrossNorm(config.hidden_size)
	def forward(self, input_embed, context, side_len_input, side_len_context):
		bs = input_embed.shape[0]
		reduce_factor = side_len_input//side_len_context

		input_embed = input_embed.view(bs, side_len_input, side_len_input+1, -1)
		context = context.view(bs, side_len_context, side_len_context+1, -1)

		input_embed = input_embed[:, :, :-1].view(bs, side_len_input, side_len_input, -1)
		input_embed = input_embed.view(bs, side_len_context, reduce_factor, side_len_context, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().flatten(0, 4)

		context_newline = context[:, :, -1:]
		context = context[:, :, :-1].view(bs, side_len_context, side_len_context, 1, 1, -1).repeat(1, 1, 1, reduce_factor, reduce_factor, 1).flatten(1, 4)

		ref_for_norm = context
		context = context.flatten(0, 1)

		context = self.context_proj(context)
		residual = input_embed
		input_embed = self.input_proj(input_embed)
		input_embed = self.layernorm_pre(torch.cat([input_embed, context], -1))
		input_embed = self.proj(input_embed) + residual
		input_embed = self.cross_norm_post(input_embed.view(bs, side_len_input*side_len_input, -1), ref_for_norm)
		
		input_embed = input_embed.view(bs, side_len_context, side_len_context, reduce_factor, reduce_factor, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, side_len_input, side_len_input, -1)

		input_embed_newline = torch.repeat_interleave(context_newline, reduce_factor, 1)

		input_embed = torch.cat([input_embed, input_embed_newline], 2).flatten(1,2)

		return input_embed


def apply_rotary_pos_emb(q, k, cos, sin, position_ids_q, position_ids_k, unsqueeze_dim=1):
	cos_q = cos[position_ids_q].unsqueeze(unsqueeze_dim)
	sin_q = sin[position_ids_q].unsqueeze(unsqueeze_dim)
	q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
	cos_k = cos[position_ids_k].unsqueeze(unsqueeze_dim)
	sin_k = sin[position_ids_k].unsqueeze(unsqueeze_dim)
	k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
	return q_embed, k_embed


# Adapted from LlamaAttention.forward
def LlamaSdpaAttention_forward(
	self,
	hidden_states,
	kv_states,
	attention_mask = None,
	position_ids_q = None,
	position_ids_kv = None,
	past_key_value = None,
	output_attentions = False,
	use_cache= False,
):

	bsz, q_len, _ = hidden_states.size()
	kv_seq_len = kv_states.shape[1]

	query_states = self.q_proj(hidden_states)
	key_states = self.k_proj(kv_states)
	value_states = self.v_proj(kv_states)

	query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
	key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
	value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

	cos, sin = self.rotary_emb(value_states, seq_len=max(q_len, kv_seq_len))

	query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids_q, position_ids_kv)

	key_states = repeat_kv(key_states, self.num_key_value_groups)
	value_states = repeat_kv(value_states, self.num_key_value_groups)

	if attention_mask is not None:
		if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
			raise ValueError(
				f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
			)

	# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
	# Reference: https://github.com/pytorch/pytorch/issues/112577.
	if query_states.device.type == "cuda" and attention_mask is not None:
		query_states = query_states.contiguous()
		key_states = key_states.contiguous()
		value_states = value_states.contiguous()

	attn_output = torch.nn.functional.scaled_dot_product_attention(
		query_states,
		key_states,
		value_states,
		attn_mask=attention_mask,
		dropout_p=self.attention_dropout if self.training else 0.0,
		# The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
		is_causal=self.is_causal and attention_mask is None and q_len > 1,
	)

	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

	attn_output = self.o_proj(attn_output)

	return attn_output, None, past_key_value

LlamaSdpaAttention.forward = LlamaSdpaAttention_forward

def decoder_forward(
	self,
	hidden_states,
	kv_states,
	attention_mask = None,
	position_ids_q = None,
	position_ids_kv = None,
	# vision_full = None, 
	# vision_concise_index = None,
	# image_token_len_per_side = None,
	# image_token_len_per_side_concise = None,
	# vision_full_attention_mask = None,
	past_key_value = None,
	output_attentions = False,
	use_cache = False,
	**kwargs,):
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		kv_states = self.input_layernorm(kv_states)

		# Cross Attention
		hidden_states, self_attn_weights, present_key_value = self.self_attn(
			hidden_states=hidden_states,
			kv_states = kv_states,
			attention_mask=attention_mask,
			position_ids_q=position_ids_q,
			position_ids_kv=position_ids_kv,
			past_key_value=past_key_value,
			output_attentions=output_attentions,
			use_cache=use_cache,
			**kwargs,
		)
		hidden_states = residual + hidden_states

		# if vision_full is not None:
		# 	vision_concise = hidden_states[:, vision_concise_index[0]:vision_concise_index[1]]
		# 	vision_full = self.vision_sampler_layers(vision_full, vision_concise, image_token_len_per_side, image_token_len_per_side_concise, vision_full_attention_mask)
		# 	hidden_states = torch.cat([hidden_states, vision_full], 1)

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if use_cache:
			outputs += (present_key_value,)

		return outputs


def decoder_forward_vision(
	self,
	hidden_states_sys,
	hidden_states_vision_concise,
	hidden_states_vision_full,
	hidden_states_text,
	attention_mask = None,
	position_ids_sys = None,
	position_ids_vision_concise = None,
	position_ids_vision_full = None,
	position_ids_vision_text = None,
	image_token_len_per_side = None,
	image_token_len_per_side_concise = None,
	vision_full_attention_mask = None,
	past_key_value = None,
	output_attentions = False,
	use_cache = False,
	**kwargs,):
		len_sys = hidden_states_sys.shape[1]
		len_vision_concise = hidden_states_vision_concise.shape[1]
		len_vision_full = hidden_states_vision_full.shape[1]
		len_text = hidden_states_text.shape[1]

		hidden_states = torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_vision_full, hidden_states_text], 1)
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		hidden_states_sys, hidden_states_vision_concise, hidden_states_vision_full, hidden_states_text = torch.split(hidden_states, [len_sys, len_vision_concise, len_vision_full, len_text], 1)

		q_states = torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_text], 1)
		kv_states = torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_vision_full, hidden_states_text], 1)
		position_ids_q = torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_text], dim=1)
		position_ids_kv = torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_full, position_ids_vision_text], dim=1)

		# Cross Attention
		q_states, self_attn_weights, present_key_value = self.self_attn(
			hidden_states=q_states,
			kv_states = kv_states,
			attention_mask=attention_mask,
			position_ids_q=position_ids_q,
			position_ids_kv=position_ids_kv,
			past_key_value=past_key_value,
			output_attentions=output_attentions,
			use_cache=use_cache,
			**kwargs,
		)

		hidden_states_sys, hidden_states_vision_concise, hidden_states_text = torch.split(q_states, [len_sys, len_vision_concise, len_text], 1)
		hidden_states_vision_full = self.vision_sampler_layers.sa(hidden_states_vision_full, hidden_states_vision_concise, image_token_len_per_side, image_token_len_per_side_concise, vision_full_attention_mask)

		hidden_states = torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_vision_full, hidden_states_text], 1)
		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states_sys, hidden_states_vision_concise, hidden_states_vision_full, hidden_states_text = torch.split(hidden_states, [len_sys, len_vision_concise, len_vision_full, len_text], 1)


		q_states = torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_text], 1)
		q_states = self.mlp(q_states)
		hidden_states_sys, hidden_states_vision_concise, hidden_states_text = torch.split(q_states, [len_sys, len_vision_concise, len_text], 1)
		hidden_states_vision_full = self.vision_sampler_layers.ffn(hidden_states_vision_full)
		
		hidden_states = torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_vision_full, hidden_states_text], 1)

		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if use_cache:
			outputs += (present_key_value,)

		return outputs

LlamaDecoderLayer.forward = decoder_forward
# LlamaDecoderLayer.forward_vision = decoder_forward_vision