import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.modeling_llama import LlamaRMSNorm


@torch.no_grad()
def svd_init(decoder_layer, mlp_layer=None):
	# Extract the original layers
	o_proj = decoder_layer.self_attn.o_proj

	# Extract the new layers
	if mlp_layer is not None:
		new_layer1 = mlp_layer.proj1
		new_layer2 = mlp_layer.proj2
	else:
		new_layer1 = decoder_layer.vision_mlp_layers.sa.proj1
		new_layer2 = decoder_layer.vision_mlp_layers.sa.proj2

	hidden_dim = new_layer1.out_features

	W_o = o_proj.weight.data

	U, S, Vh = torch.linalg.svd(W_o.float(), full_matrices=False)
	U = U.to(W_o.dtype)
	S = S.to(W_o.dtype)
	Vh = Vh.to(W_o.dtype)
	U_prime = U[:, :hidden_dim]
	S_prime = S[:hidden_dim]
	Vh_prime = Vh[:hidden_dim, :]

	S_sqrt = torch.diag(torch.sqrt(S_prime + 1e-6))

	W1 = torch.matmul(S_sqrt, Vh_prime)
	W2 = torch.matmul(U_prime, S_sqrt)

	new_layer1.weight.copy_(W1)
	new_layer2.weight.copy_(W2)

# @torch.no_grad()
# def svd_init(decoder_layer, mlp_layer=None, bias=False):
# 	# Extract the original layers
# 	v_proj = decoder_layer.self_attn.v_proj
# 	o_proj = decoder_layer.self_attn.o_proj

# 	# Extract the new layers
# 	if mlp_layer is not None:
# 		new_layer1 = mlp_layer.proj1
# 		new_layer2 = mlp_layer.proj2
# 	else:
# 		new_layer1 = decoder_layer.vision_mlp_layers.sa.proj1
# 		new_layer2 = decoder_layer.vision_mlp_layers.sa.proj2

# 	dim2 = new_layer1.out_features

# 	# Extract weights and biases
# 	W_v = v_proj.weight.data

# 	num_key_value_groups = decoder_layer.self_attn.num_key_value_groups
# 	n_rep = num_key_value_groups
# 	num_key_value_heads = decoder_layer.self_attn.num_key_value_heads
# 	head_dim = decoder_layer.self_attn.head_dim
# 	W_v_repeat = []
# 	if bias:
# 		b_v = v_proj.bias.data
# 		b_v_repeat = []
# 	for i in range(num_key_value_heads):
# 		W_v_repeat.append(W_v[head_dim*i:head_dim*(i+1)].repeat(n_rep, 1))
# 		if bias:
# 			b_v_repeat.append(b_v[head_dim*i:head_dim*(i+1)].repeat(n_rep))
# 	W_v_repeat = torch.cat(W_v_repeat, 0)
# 	W_v = W_v_repeat
# 	if bias:
# 		b_v_repeat = torch.cat(b_v_repeat)
# 		b_v = b_v_repeat
	
# 	W_o = o_proj.weight.data

# 	# Compute the combined weight matrix and bias vector
# 	M = torch.matmul(W_o, W_v)
# 	if bias:
# 		b = torch.matmul(b_v, W_o.t())

# 	# Perform truncated SVD on the combined weight matrix
# 	U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
# 	U = U.to(W_o.dtype)
# 	S = S.to(W_o.dtype)
# 	Vh = Vh.to(W_o.dtype)
# 	U_prime = U[:, :dim2]
# 	S_prime = S[:dim2]
# 	Vh_prime = Vh[:dim2, :]

# 	# Compute the square root of the singular values matrix
# 	S_sqrt = torch.diag(torch.sqrt(S_prime + 1e-6))

# 	# Initialize the new weights
# 	W1 = torch.matmul(S_sqrt, Vh_prime)
# 	W2 = torch.matmul(U_prime, S_sqrt)

# 	# Assign the computed weights and bias to the new layers
# 	new_layer1.weight.copy_(W1)
# 	new_layer2.weight.copy_(W2)
# 	if bias:
# 		new_layer2.bias.copy_(b)


# class VisionMLP(nn.Module):
# 	def __init__(self, config, intermediate_size, bias=False):
# 		super().__init__()
# 		self.proj1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		self.proj2 = nn.Linear(intermediate_size, config.hidden_size, bias=bias)
# 		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		# self.gate = nn.Sequential(nn.Linear(intermediate_size, intermediate_size, bias=False), nn.Sigmoid())
# 		self.proj = nn.Sequential(
# 			nn.Linear(intermediate_size*2, intermediate_size, bias=False),
# 			nn.SiLU(),
# 			nn.Linear(intermediate_size, config.hidden_size, bias=False)
# 		)
# 		self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# 	def forward(self, image_full, image_compress, compress_reduce_factor, per_crop_token_len=576, attention_mask=None):
# 		image_full = self.proj2(self.proj1(image_full))
# 		side_len_full = int(per_crop_token_len**0.5)
# 		side_len_compress = side_len_full // compress_reduce_factor

# 		num_image_crops = image_full.shape[1]//per_crop_token_len
# 		bs = image_full.shape[0]

# 		image_full = image_full.view(bs*num_image_crops, side_len_full, side_len_full, -1)
# 		image_compress = image_compress.view(bs*num_image_crops, side_len_compress, side_len_compress, -1)
# 		image_compress = self.context_proj(image_compress)
# 		image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2)
# 		residual = image_full
# 		image_full = self.input_proj(image_full)
# 		image_full = torch.cat([image_full, image_compress], -1)
# 		# comb_weight = self.gate(image_full + image_compress)
# 		# image_full = comb_weight * image_full + (1 - comb_weight) * image_compress
# 		image_full = self.layernorm_post(self.proj(image_full) + residual) 

# 		image_full = image_full.view(bs, num_image_crops*side_len_full*side_len_full, -1)

# 		return image_full


# class VisionMLP(nn.Module):
# 	def __init__(self, config, intermediate_size):
# 		super().__init__()
# 		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
# 		# self.gate = nn.Sequential(nn.Linear(intermediate_size, intermediate_size, bias=False), nn.Sigmoid())
# 		self.proj = nn.Sequential(
# 			nn.Linear(intermediate_size*2, intermediate_size, bias=False),
# 			nn.SiLU(),
# 			nn.Linear(intermediate_size, config.hidden_size, bias=False)
# 		)
# 		self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# 	def forward(self, image_full, image_compress, compress_reduce_factor, per_crop_token_len=576, attention_mask=None):
# 		side_len_full = int(per_crop_token_len**0.5)
# 		side_len_compress = side_len_full // compress_reduce_factor

# 		num_image_crops = image_full.shape[1]//per_crop_token_len
# 		bs = image_full.shape[0]

# 		image_full = image_full.view(bs*num_image_crops, side_len_full, side_len_full, -1)
# 		image_compress = image_compress.view(bs*num_image_crops, side_len_compress, side_len_compress, -1).contiguous()
# 		image_compress = self.context_proj(image_compress)
# 		image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2)
# 		residual = image_full
# 		image_full = self.input_proj(image_full)
# 		image_full = torch.cat([image_full, image_compress], -1)
# 		# comb_weight = self.gate(image_full + image_compress)
# 		# image_full = comb_weight * image_full + (1 - comb_weight) * image_compress
# 		image_full = self.layernorm_post(self.proj(image_full) + residual) 

# 		image_full = image_full.view(bs, num_image_crops*side_len_full*side_len_full, -1)

# 		return image_full


# class VisionMLP_sa(nn.Module):
# 	def __init__(self, config, intermediate_size=1024):
# 		super().__init__()
		# self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		# self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		# self.proj = nn.Sequential(
		# 	nn.Linear(intermediate_size*2, intermediate_size, bias=False),
		# 	nn.SiLU(),
		# 	nn.Linear(intermediate_size, config.hidden_size, bias=False)
		# )

# 	def forward(self, image_full, image_compress, compress_reduce_factor, per_crop_token_len=576, attention_mask=None):
# 		side_len_full = int(per_crop_token_len**0.5)
# 		side_len_compress = side_len_full // compress_reduce_factor

# 		num_image_crops = image_full.shape[1]//per_crop_token_len
# 		bs = image_full.shape[0]

# 		image_full = image_full.view(bs*num_image_crops, side_len_full, side_len_full, -1)
# 		image_compress = image_compress.view(bs*num_image_crops, side_len_compress, side_len_compress, -1)
# 		image_compress = self.context_proj(image_compress)
# 		image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2)
# 		residual = image_full
# 		image_full = self.input_proj(image_full)
# 		image_full = torch.cat([image_full, image_compress], -1)
# 		# image_full = self.layernorm_post(self.proj(image_full) + residual) 
# 		image_full = self.proj(image_full)

# 		image_full = image_full.view(bs, num_image_crops*side_len_full*side_len_full, -1)

# 		return image_full
	
class VisionMLP_sa(nn.Module):
	def __init__(self, config, intermediate_size=1024):
		super().__init__()
		# self.proj1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		# self.proj2 = nn.Linear(intermediate_size, config.hidden_size, bias=False)

		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.hidden_size // self.num_heads


		# self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.proj = nn.Sequential(
			nn.Linear(intermediate_size, intermediate_size, bias=False),
			nn.SiLU(),
			nn.Linear(intermediate_size, config.hidden_size, bias=False)
		)


		# self.pool = nn.AvgPool2d(kernel_size=config.compress_reduce_factor, stride=config.compress_reduce_factor)
		# self.gate = nn.Sequential(
		# 	nn.Linear(self.head_dim, 3, bias=False),
		# 	nn.Softmax(-1)
		# )

		# self.gate = nn.Sequential(
		# 	nn.Linear(self.head_dim, 1, bias=False),
		# 	nn.Sigmoid(),
		# )

		# self.gate = nn.Sequential(
		# 	nn.Linear(config.hidden_size, config.hidden_size, bias=False),
		# 	nn.Sigmoid(),
		# )

	def forward(self, image_full, image_compress=None, compress_reduce_factor=None, per_crop_token_len=576, attention_mask=None):
		# return image_full.transpose(1, 2).contiguous().flatten(2,3)
		# side_len_full = int(per_crop_token_len**0.5)
		# side_len_compress = side_len_full // compress_reduce_factor

		# num_image_crops = image_full.shape[2]//per_crop_token_len
		# bs = image_full.shape[0]

		# # image_compress = image_compress.view(bs*self.num_heads*num_image_crops, side_len_compress, side_len_compress, -1)
		# # image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2).view(bs, self.num_heads, num_image_crops*side_len_full*side_len_full, -1)

		# # gate_weight = self.gate(image_full)
		# # image_full = gate_weight * image_full + (1-gate_weight) * image_compress
		# image_full = image_full.transpose(1, 2).contiguous().flatten(2,3)
		# image_full = self.proj1(image_full)
		# image_full = self.proj2(image_full)
		# return image_full

		side_len_full = int(per_crop_token_len**0.5)
		side_len_compress = side_len_full // compress_reduce_factor

		num_image_crops = image_full.shape[1]//per_crop_token_len
		bs = image_full.shape[0]

		# image_compress = image_compress.view(bs*num_image_crops, side_len_compress, side_len_compress, -1)
		# image_compress = self.context_proj(image_compress)
		# image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2).view(bs, num_image_crops*side_len_full*side_len_full, -1)

		image_full = self.input_proj(image_full)
		# image_full = self.proj(torch.cat([image_full, image_compress], -1))
		image_full = self.proj(image_full)

		return image_full
	

class VisionMLP_ffn(nn.Module):
	def __init__(self, config, intermediate_size=1024):
		super().__init__()
		# self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		# self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

		# self.proj = nn.Sequential(
		# 	nn.Linear(config.hidden_size, intermediate_size, bias=False),
		# 	nn.SiLU(),
		# 	nn.Linear(intermediate_size, config.hidden_size, bias=False)
		# )

		# self.proj = nn.Sequential(
		# 	nn.Linear(intermediate_size*2, intermediate_size, bias=False),
		# 	nn.SiLU(),
		# 	nn.Linear(intermediate_size, config.hidden_size, bias=False)
		# )


		# self.proj1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		# self.proj2 = nn.Sequential(
		# 	nn.SiLU(),
		# 	nn.Linear(intermediate_size, config.hidden_size, bias=False)
		# )

		# self.gate = nn.Sequential(
		# 	nn.Linear(intermediate_size, config.hidden_size, bias=False),
		# 	nn.Sigmoid(),
		# )

		# self.proj = nn.Sequential(
		# 	nn.Linear(config.hidden_size, intermediate_size, bias=False),
		# 	nn.SiLU(),
		# 	nn.Linear(intermediate_size, config.hidden_size, bias=False)
		# )

	# def forward(self, image_full, image_compress = None, compress_reduce_factor=4, per_crop_token_len=576, attention_mask=None):
	# 	image_full = self.input_proj(image_full)
	# 	image_compress = self.context_proj(image_compress)

	# 	side_len_full = int(per_crop_token_len**0.5)
	# 	side_len_compress = side_len_full // compress_reduce_factor
	# 	num_image_crops = image_full.shape[1]//per_crop_token_len
	# 	bs = image_full.shape[0]
	# 	image_compress = image_compress.view(bs*num_image_crops, side_len_compress, side_len_compress, -1)
	# 	image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2).view(bs, num_image_crops*side_len_full*side_len_full, -1)

	# 	image_full = self.proj(torch.cat([image_full, image_compress], -1))
	# 	return image_full

	def forward(self, image_full, image_compress = None, compress_reduce_factor=4, per_crop_token_len=576, attention_mask=None):
		return image_full
		image_full = self.proj(image_full).to(image_full.dtype)

		return image_full
	
class VisionMLP(nn.Module):
	def __init__(self, config, intermediate_size=1024, bias=False):
		super().__init__()
		self.sa = VisionMLP_sa(config, intermediate_size)
		# self.sa = nn.Identity()
		self.ffn = VisionMLP_ffn(config, intermediate_size)


	# def forward(self, image_full, image_compress, compress_reduce_factor, per_crop_token_len=576, attention_mask=None):
	# 	side_len_full = int(per_crop_token_len**0.5)
	# 	side_len_compress = side_len_full // compress_reduce_factor

	# 	num_image_crops = image_full.shape[1]//per_crop_token_len
	# 	bs = image_full.shape[0]

	# 	image_full = self.image

	# 	image_full = image_full.view(bs*num_image_crops, side_len_full, side_len_full, -1)
	# 	image_compress = image_compress.view(bs*num_image_crops, side_len_compress, side_len_compress, -1)
	# 	image_compress = self.context_proj(image_compress)
	# 	image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2)
	# 	residual = image_full
	# 	image_full = self.input_proj(image_full)
	# 	image_full = torch.cat([image_full, image_compress], -1)
	# 	# comb_weight = self.gate(image_full + image_compress)
	# 	# image_full = comb_weight * image_full + (1 - comb_weight) * image_compress
	# 	image_full = self.layernorm_post(self.proj(image_full) + residual) 

	# 	image_full = image_full.view(bs, num_image_crops*side_len_full*side_len_full, -1)

	# 	return image_full