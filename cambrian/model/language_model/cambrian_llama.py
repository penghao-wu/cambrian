#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, \
						 LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import logging

from transformers.modeling_attn_mask_utils import (
	AttentionMaskConverter,
	_prepare_4d_attention_mask,
	_prepare_4d_causal_attention_mask,
	_prepare_4d_causal_attention_mask_for_sdpa,
)

from ..cambrian_arch import CambrianMetaModel, CambrianMetaForCausalLM
from cambrian.utils import IS_XLA_AVAILABLE

logger = logging.get_logger(__name__)

from dataclasses import dataclass
@dataclass
class CausalLMOutputWithPastWithAuxLoss(CausalLMOutputWithPast):
	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None
	past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
	hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	attentions: Optional[Tuple[torch.FloatTensor]] = None
	lm_loss: Optional[torch.FloatTensor] = None
	aux_loss: Optional[torch.FloatTensor] = None


@dataclass
class BaseModelOutputWithPastWithAuxLoss(BaseModelOutputWithPast):
	last_hidden_state: torch.FloatTensor = None
	past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
	hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	attentions: Optional[Tuple[torch.FloatTensor]] = None
	aux_loss: Optional[torch.FloatTensor] = None

class CambrianConfig(LlamaConfig):
	model_type = "cambrian_llama"

	debug = "debug"

def get_image_compress(hidden_states_image_full, compress_reduce_factor, per_crop_token_len=576):
	bs = hidden_states_image_full.shape[0]
	num_image_crops = hidden_states_image_full.shape[1]//per_crop_token_len
	h_full = w_full = int(per_crop_token_len**0.5)
	h_compress = w_compress = h_full//compress_reduce_factor

	hidden_states_image_full = hidden_states_image_full.view(bs*num_image_crops, h_full, w_full, -1)
	
	hidden_states_image_compress = nn.functional.interpolate(
	hidden_states_image_full.permute(0, 3, 1, 2).contiguous(),
		size=(h_compress, w_compress),
		mode='bilinear',
		align_corners=False
	)
	hidden_states_image_compress = hidden_states_image_compress.permute(0, 2, 3, 1).contiguous().view(bs, num_image_crops*h_compress*w_compress, -1)
	return hidden_states_image_compress

class CambrianLlamaModel(CambrianMetaModel, LlamaModel):
	config_class = CambrianConfig

	def __init__(self, config: LlamaConfig):
		super(CambrianLlamaModel, self).__init__(config)

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		attention_mask_regular_4d: Optional[torch.Tensor] = None,
		attention_mask_compress_4d: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		position_ids_image_compress: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, BaseModelOutputWithPast]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		self._use_flash_attention_2 = getattr(self, '_use_flash_attention_2', False)
		self._use_sdpa = getattr(self, '_use_sdpa', True)

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = None

		max_num_image_crops = self.config.max_num_image_crops
		per_crop_token_len = self.config.per_crop_token_len
		compress_reduce_factor = self.config.compress_reduce_factor
		compress_v = self.config.compress_v
		compress_v_start_layer = self.config.compress_v_start_layer

		image_full_len = max_num_image_crops * per_crop_token_len
		newline_full_len = max_num_image_crops
		image_compress_len = max_num_image_crops * per_crop_token_len // compress_reduce_factor**2
		text_len = inputs_embeds.shape[1] - image_full_len - newline_full_len

		hidden_states = inputs_embeds

		hidden_states_image_full = hidden_states[:, :image_full_len]
		hidden_states_newline_full = hidden_states[:, image_full_len:image_full_len+newline_full_len]
		hidden_states_text = hidden_states[:, image_full_len+newline_full_len:]

		aux_loss_total = 0
		for layer_i, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			if not compress_v or layer_i < compress_v_start_layer:

				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						hidden_states,
						attention_mask_regular_4d if layer_i < 100 else attention_mask_compress_4d,
						position_ids,
						position_ids,
						past_key_values,
						output_attentions,
						use_cache,
						False,
						image_compress_len,
						image_full_len
					)
				else:
					layer_outputs = decoder_layer(
						hidden_states,
						attention_mask_regular_4d if layer_i < 100 else attention_mask_compress_4d,
						position_ids,
						position_ids,
						past_key_values,
						output_attentions,
						use_cache,
						False,
						image_compress_len,
						image_full_len
					)
				
				hidden_states = layer_outputs[0]
							
			else:
				if layer_i == compress_v_start_layer:
					hidden_states_image_full = hidden_states[:, :image_full_len]
					hidden_states_newline_full = hidden_states[:, image_full_len:image_full_len+newline_full_len]
					hidden_states_text = hidden_states[:, image_full_len+newline_full_len:]

					position_ids_image_full = position_ids[:, :image_full_len]
					position_ids_newline_full = position_ids[:, image_full_len:image_full_len+newline_full_len]
					position_ids_text = position_ids[:, image_full_len+newline_full_len:]

					hidden_states_image_compress = get_image_compress(hidden_states_image_full, compress_reduce_factor, per_crop_token_len)

					# position_ids_compress_q = torch.cat([position_ids_newline_full, position_ids_text], 1)
					# position_ids_compress_kv = torch.cat([position_ids_image_full,  position_ids_newline_full, position_ids_text], 1)

					# position_ids_compress_q = torch.cat([position_ids_newline_full, position_ids_text], 1)
					position_ids_compress_q = torch.cat([position_ids_image_compress, position_ids_newline_full, position_ids_text], 1)
					position_ids_compress_kv = torch.cat([position_ids_image_full, position_ids_image_compress,  position_ids_newline_full, position_ids_text], 1)
					# attention_mask_compress_4d = attention_mask_regular_4d[:, :, image_full_len:]

				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						torch.cat([hidden_states_image_full, hidden_states_image_compress, hidden_states_newline_full, hidden_states_text], 1),
						# torch.cat([hidden_states_image_full, hidden_states_newline_full, hidden_states_text], 1),
						attention_mask_compress_4d,
						position_ids_compress_q,
						position_ids_compress_kv,
						past_key_values,
						output_attentions,
						use_cache,
						True,
						image_compress_len,
						image_full_len
					)
				else:
					layer_outputs = decoder_layer(
						torch.cat([hidden_states_image_full, hidden_states_image_compress, hidden_states_newline_full, hidden_states_text], 1),
						# torch.cat([hidden_states_image_full, hidden_states_newline_full, hidden_states_text], 1),
						attention_mask_compress_4d,
						position_ids_compress_q,
						position_ids_compress_kv,
						past_key_values,
						output_attentions,
						use_cache,
						True,
						image_compress_len,
						image_full_len
					)

				# hidden_states_image_full = layer_outputs[0][:, :image_full_len]
				# hidden_states_image_compress = layer_outputs[0][:, image_full_len:image_full_len+image_compress_len]
				# hidden_states_newline_full = layer_outputs[0][:, image_full_len+image_compress_len:image_full_len+image_compress_len+newline_full_len]
				# hidden_states_text = layer_outputs[0][:, image_full_len+image_compress_len+newline_full_len:]

				hidden_states_image_compress = layer_outputs[0][:, :image_compress_len]
				hidden_states_newline_full = layer_outputs[0][:, image_compress_len:image_compress_len+newline_full_len]
				hidden_states_text = layer_outputs[0][:, image_compress_len+newline_full_len:]
				# hidden_states_image_full = decoder_layer.vision_mlp_layers(hidden_states_image_full, hidden_states_image_compress, compress_reduce_factor, per_crop_token_len)
				hidden_states_image_full = self.vision_mlp_layers[layer_i-compress_v_start_layer](hidden_states_image_full, hidden_states_image_compress, compress_reduce_factor, per_crop_token_len)
				
				aux_loss = 0
				aux_loss_total += aux_loss/self.config.num_of_vision_mlp_layers
				if layer_i == len(self.layers) - 1:
					hidden_states = torch.cat([hidden_states_image_full, hidden_states_newline_full, hidden_states_text], 1)

		hidden_states = self.norm(hidden_states)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		next_cache = None
		if not return_dict:
			return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, aux_loss_total] if v is not None)
		return BaseModelOutputWithPastWithAuxLoss(
			last_hidden_state=hidden_states,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
			aux_loss=aux_loss_total,
		)


class CambrianLlamaForCausalLM(LlamaForCausalLM, CambrianMetaForCausalLM):
	config_class = CambrianConfig

	def __init__(self, config):
		super(LlamaForCausalLM, self).__init__(config)

		self.model = CambrianLlamaModel(config)
		self.pretraining_tp = config.pretraining_tp
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

		# Initialize weights and apply final processing
		self.post_init()

	def get_model(self):
		return self.model

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		attention_mask_regular_4d: Optional[torch.Tensor] = None,
		attention_mask_compress_4d: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		position_ids_image_compress: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		images: Optional[torch.FloatTensor] = None,
		image_sizes: Optional[List[List[int]]] = None,
		return_dict: Optional[bool] = None,
		cache_position = None
	) -> Union[Tuple, CausalLMOutputWithPast]:

		if inputs_embeds is None:
			(
				input_ids,
				inputs_embeds,
			) = self.prepare_inputs_labels_for_multimodal(
				input_ids,
				images,
			)
		if IS_XLA_AVAILABLE:
			# Very Important for TorchXLA
			#self.model.gradient_checkpointing = False
				
			from torch_xla.utils.checkpoint import checkpoint
			# self.model.gradient_checkpointing = True
			self.model._gradient_checkpointing_func = checkpoint

		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# training
		if IS_XLA_AVAILABLE:
			# decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
			outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			attention_mask_regular_4d=attention_mask_regular_4d,
			attention_mask_compress_4d=attention_mask_compress_4d,
			position_ids=position_ids,
			position_ids_image_compress=position_ids_image_compress,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		# inference
		else:
			if hasattr(self, "vision_tower_aux_feature_list"):
			# decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
				outputs = self.model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					position_ids=position_ids,
					past_key_values=past_key_values,
					inputs_embeds=inputs_embeds,
					use_cache=use_cache,
					output_attentions=output_attentions,
					output_hidden_states=output_hidden_states,
					return_dict=return_dict,
					vision_tower_aux_feature_list=vision_tower_aux_feature_list if inputs_embeds is None else self.vision_tower_aux_feature_list,
					vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list if inputs_embeds is None else self.vision_tower_aux_attention_masks_list, 
					final_vision_feature_size=final_vision_feature_size if inputs_embeds is None else self.final_vision_feature_size,
					global_context_feature=global_context_feature if inputs_embeds is None else self.global_context_feature,
				)
			else:
				outputs = self.model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					position_ids=position_ids,
					past_key_values=past_key_values,
					inputs_embeds=inputs_embeds,
					use_cache=use_cache,
					output_attentions=output_attentions,
					output_hidden_states=output_hidden_states,
					return_dict=return_dict,
				)

		hidden_states = outputs[0]
		if self.config.pretraining_tp > 1:
			lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
			logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
			logits = torch.cat(logits, dim=-1)
		else:
			logits = self.lm_head(hidden_states)
		logits = logits.float()

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			# shift_logits = logits[..., :-1, :].contiguous()
			# shift_labels = labels[..., 1:].contiguous()
			shift_logits = logits
			shift_labels = labels
			# Flatten the tokens
			loss_fct = CrossEntropyLoss()
			shift_logits = shift_logits.view(-1, self.config.vocab_size)
			shift_labels = shift_labels.view(-1)
			# Enable model parallelism
			shift_labels = shift_labels.to(shift_logits.device)
			loss = loss_fct(shift_logits, shift_labels)

		if not return_dict:
			output = (logits,) + outputs[1:]
			return (loss,) + output if loss is not None else output

		return CausalLMOutputWithPast(
			loss=loss,
			logits=logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


	@torch.no_grad()
	def generate(
		self,
		inputs: Optional[torch.Tensor] = None,
		images: Optional[torch.Tensor] = None,
		image_sizes: Optional[torch.Tensor] = None,
		**kwargs,
	) -> Union[GenerateOutput, torch.LongTensor]:
		position_ids = kwargs.pop("position_ids", None)
		attention_mask = kwargs.pop("attention_mask", None)
		if "inputs_embeds" in kwargs:
			raise NotImplementedError("`inputs_embeds` is not supported")

		if images is not None:
			(
				inputs,
				position_ids,
				attention_mask,
				_,
				inputs_embeds,
				_,
				vision_tower_aux_feature_list,
				vision_tower_aux_attention_masks_list,
				final_vision_feature_size,
				global_context_feature,
			) = self.prepare_inputs_labels_for_multimodal(
				inputs,
				position_ids,
				attention_mask,
				None,
				None,
				images,
				image_sizes=image_sizes
			)
			self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
			self.vision_tower_aux_attention_masks_list = vision_tower_aux_attention_masks_list
			self.final_vision_feature_size = final_vision_feature_size
			self.global_context_feature = global_context_feature
		else:
			inputs_embeds = self.get_model().embed_tokens(inputs)

		return super().generate(
			position_ids=position_ids,
			attention_mask=attention_mask,
			inputs_embeds=inputs_embeds,
			**kwargs
		)

	def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
									  inputs_embeds=None, **kwargs):
		images = kwargs.pop("images", None)
		image_sizes = kwargs.pop("image_sizes", None)
		inputs = super().prepare_inputs_for_generation(
			input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
		)
		if images is not None:
			inputs['images'] = images
		if image_sizes is not None:
			inputs['image_sizes'] = image_sizes
		return inputs
	
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaDecoderLayer, LlamaRMSNorm, rotate_half, repeat_kv
def apply_rotary_pos_emb(q, k, cos, sin, position_ids_q, position_ids_k, unsqueeze_dim=1):
	cos_q = cos[position_ids_q].unsqueeze(unsqueeze_dim)
	sin_q = sin[position_ids_q].unsqueeze(unsqueeze_dim)
	q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
	cos_k = cos[position_ids_k].unsqueeze(unsqueeze_dim)
	sin_k = sin[position_ids_k].unsqueeze(unsqueeze_dim)
	k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
	return q_embed, k_embed

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
	sep_sa = False,
	image_compress_len=36,
	image_full_len=576,
	vision_mlp=None,
):

	bsz, q_len, _ = hidden_states.size()
	kv_seq_len = kv_states.shape[1]

	query_states = self.q_proj(hidden_states).to(hidden_states.dtype)
	key_states = self.k_proj(kv_states).to(hidden_states.dtype)
	value_states = self.v_proj(kv_states).to(hidden_states.dtype)

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

	# if sep_sa:
	# 	value_states_image_full = value_states[:, :, :image_full_len]
	# 	value_states_image_compress = attn_output[:, :, :image_compress_len]
	# 	value_states_image_full = vision_mlp.sa(value_states_image_full, value_states_image_compress, int((image_full_len//image_compress_len)**0.5), image_full_len)

	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

	# if sep_sa:
	# 	attn_output = torch.cat([value_states_image_full, attn_output], 1)

	attn_output = self.o_proj(attn_output)

	if sep_sa:
		kv_states_image_full = kv_states[:, :image_full_len]
		output_image_compress = attn_output[:, :image_compress_len]
		value_states_image_full = vision_mlp.sa(kv_states_image_full, output_image_compress, int((image_full_len//image_compress_len)**0.5), image_full_len)
		attn_output = torch.cat([value_states_image_full, attn_output], 1)

	# if sep_sa:
	# 	attn_output = torch.cat([value_states_image_full, attn_output], 1)

	return attn_output, None, past_key_value
# def LlamaSdpaAttention_forward(
# 	self,
# 	hidden_states,
# 	kv_states,
# 	attention_mask = None,
# 	position_ids_q = None,
# 	position_ids_kv = None,
# 	past_key_value = None,
# 	output_attentions = False,
# 	use_cache= False,
# ):

# 	bsz, q_len, _ = hidden_states.size()
# 	kv_seq_len = kv_states.shape[1]

# 	query_states = self.q_proj(hidden_states).to(hidden_states.dtype)
# 	key_states = self.k_proj(kv_states).to(hidden_states.dtype)
# 	value_states = self.v_proj(kv_states).to(hidden_states.dtype)

# 	query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
# 	key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
# 	value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

# 	cos, sin = self.rotary_emb(value_states, seq_len=max(q_len, kv_seq_len))

# 	query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids_q, position_ids_kv)

# 	key_states = repeat_kv(key_states, self.num_key_value_groups)
# 	value_states = repeat_kv(value_states, self.num_key_value_groups)

# 	if attention_mask is not None:
# 		if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
# 			raise ValueError(
# 				f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
# 			)

# 	# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
# 	# Reference: https://github.com/pytorch/pytorch/issues/112577.
# 	if query_states.device.type == "cuda" and attention_mask is not None:
# 		query_states = query_states.contiguous()
# 		key_states = key_states.contiguous()
# 		value_states = value_states.contiguous()

# 	attn_output = torch.nn.functional.scaled_dot_product_attention(
# 		query_states,
# 		key_states,
# 		value_states,
# 		attn_mask=attention_mask,
# 		dropout_p=self.attention_dropout if self.training else 0.0,
# 		# The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
# 		is_causal=self.is_causal and attention_mask is None and q_len > 1,
# 	)

# 	attn_output = attn_output.transpose(1, 2).contiguous()
# 	attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

# 	attn_output = self.o_proj(attn_output)

# 	return attn_output, None, past_key_value

LlamaSdpaAttention.forward = LlamaSdpaAttention_forward

# sep sa only
# def decoder_forward(
# 	self,
# 	hidden_states,
# 	attention_mask = None,
# 	position_ids_q = None,
# 	position_ids_kv = None,
# 	past_key_value = None,
# 	output_attentions = False,
# 	use_cache = False,
# 	sep_sa_ffn = False,
# 	image_compress_len=36,
# 	image_full_len=576,
# 	**kwargs,):
# 		if sep_sa_ffn:
# 			residual = hidden_states
# 			hidden_states = self.input_layernorm(hidden_states)
# 			kv_states = hidden_states
# 			hidden_states = hidden_states[:, image_full_len:]
# 		else:
# 			residual = hidden_states
# 			hidden_states = self.input_layernorm(hidden_states)
# 			kv_states = hidden_states

# 		# Cross Attention
# 		hidden_states, self_attn_weights, present_key_value = self.self_attn(
# 			hidden_states=hidden_states,
# 			kv_states = kv_states,
# 			attention_mask=attention_mask,
# 			position_ids_q=position_ids_q,
# 			position_ids_kv=position_ids_kv,
# 			past_key_value=past_key_value,
# 			output_attentions=output_attentions,
# 			use_cache=use_cache,
# 			sep_sa = sep_sa_ffn,
# 			image_compress_len=image_compress_len,
# 			image_full_len=image_full_len,
# 			vision_mlp=self.vision_mlp_layers if sep_sa_ffn else None,
# 			**kwargs,
# 		)
# 		hidden_states = residual + hidden_states

# 		# Fully Connected
# 		residual = hidden_states
# 		hidden_states = self.post_attention_layernorm(hidden_states)
# 		hidden_states = self.mlp(hidden_states)
# 		hidden_states = residual + hidden_states

# 		outputs = (hidden_states,)

# 		if output_attentions:
# 			outputs += (self_attn_weights,)

# 		if use_cache:
# 			outputs += (present_key_value,)

# 		return outputs

def decoder_forward(
	self,
	hidden_states,
	attention_mask = None,
	position_ids_q = None,
	position_ids_kv = None,
	past_key_value = None,
	output_attentions = False,
	use_cache = False,
	sep_sa_ffn=False,
	image_compress_len=36,
	image_full_len=576,
	**kwargs,):
		if sep_sa_ffn:
			residual = hidden_states[:, image_full_len:]
			hidden_states = self.input_layernorm(hidden_states)
			kv_states = hidden_states
			hidden_states = hidden_states[:, image_full_len:]
		else:
			residual = hidden_states
			hidden_states = self.input_layernorm(hidden_states)
			kv_states = hidden_states

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

LlamaDecoderLayer.forward = decoder_forward

AutoConfig.register("cambrian_llama", CambrianConfig)
AutoModelForCausalLM.register(CambrianConfig, CambrianLlamaForCausalLM)
