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

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

logger = logging.get_logger(__name__)
class CambrianConfig(Qwen2Config):
	model_type = "cambrian_qwen"

	debug = "debug"


class CambrianQwenModel(CambrianMetaModel, Qwen2Model):
	config_class = CambrianConfig

	def __init__(self, config: Qwen2Config):
		super(CambrianQwenModel, self).__init__(config)

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_masks: Optional[torch.Tensor] = None,
		attention_mask_c2f: Optional[torch.Tensor] = None,
		attention_masks_all2all: Optional[torch.Tensor] = None,
		image_valid_mask: Optional[torch.Tensor] = None,
		vision_full_attention_mask: Optional[torch.Tensor] = None,
		position_ids_sys: Optional[torch.LongTensor] = None,
		position_ids_vision_concise: Optional[torch.LongTensor] = None,
		position_ids_vision_full: Optional[torch.LongTensor] = None,
		position_ids_vision_text: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		inputs_embeds_vision_concise: Optional[torch.FloatTensor] = None,
		vision_tower_aux_feature_list: Optional[List[torch.FloatTensor]] = None,
		vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
		final_vision_feature_size: Optional[List[tuple]] = None,
		global_context_feature: Optional[torch.Tensor] = None,
		gist_token_positions: Optional[torch.LongTensor] = None,
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

		vision_token_start_idx = self.config.image_position
		
		image_token_len_per_side = int(self.config.image_token_len**0.5)
		image_token_newline_num = self.config.image_token_len + image_token_len_per_side

		image_token_len_per_side_concise = int(self.config.image_token_len_concise**0.5)
		image_token_concise_newline_num = self.config.image_token_len_concise + image_token_len_per_side_concise

		hidden_states_sys = inputs_embeds[:, :vision_token_start_idx]
		hidden_states_text = inputs_embeds[:, vision_token_start_idx+image_token_newline_num:]
		hidden_states_vision_full = inputs_embeds[:, vision_token_start_idx:vision_token_start_idx+image_token_newline_num]
		hidden_states_vision_concise = inputs_embeds_vision_concise

		len_sys = hidden_states_sys.shape[1]
		len_vision_concise = hidden_states_vision_concise.shape[1]
		len_vision_full = hidden_states_vision_full.shape[1]
		len_text = hidden_states_text.shape[1]

		hidden_states = inputs_embeds

		skip_layers = [_ for _ in range(15, 32)]
		skip_layers += [0,1,2,3,4]
		skip_layers = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
		skip_layers = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
		skip_layers = [0, 1, 2, 3 , 4, 5, 6, 7, 8, 9, 10]

		skip_layers = [12, 15, 18, 21, 24, 27, 30]
		skip_layers = [_ for _ in range(0, 32)]
		# skip_layers = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
		# skip_layers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
		# skip_layers = [_ for _ in range(0, 32)]
		skip_layers = [_ for _ in range(16, 28)]
		# skip_layers += [0, 1, 2, 3, 4, 5]

		# skip_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
		aux_loss_total = 0
		for i, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states_text,)

			if i in skip_layers:

				# [sys, vision_concise, text] to [sys, vision_full, text]

				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_text], dim=1),
						torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_vision_full, hidden_states_text], dim=1),
						attention_mask_c2f,
						torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_text], dim=1),
						torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_full, position_ids_vision_text], dim=1),
						past_key_values,
						output_attentions,
						use_cache,
					)
				else:
					layer_outputs = decoder_layer(
						torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_text], dim=1),
						torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_vision_full, hidden_states_text], dim=1),
						attention_mask_c2f,
						torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_text], dim=1),
						torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_full, position_ids_vision_text], dim=1),
						past_key_values,
						output_attentions,
						use_cache,
					)
				hidden_states_vision_concise = layer_outputs[0][:, vision_token_start_idx:vision_token_start_idx+image_token_concise_newline_num]
				# hidden_states_text = layer_outputs[0][:, vision_token_start_idx+image_token_concise_newline_num:]
				hidden_states_text = layer_outputs[0][:, vision_token_start_idx+image_token_concise_newline_num:vision_token_start_idx+image_token_concise_newline_num+len_text]
				hidden_states_sys = layer_outputs[0][:, :vision_token_start_idx]

				# hidden_states_vision_full = layer_outputs[0][:, vision_token_start_idx+image_token_concise_newline_num+len_text:]
				hidden_states_vision_full = self.vision_sampler_layers[i](hidden_states_vision_full, hidden_states_vision_concise, image_token_len_per_side, image_token_len_per_side_concise, vision_full_attention_mask)

				aux_loss_total = 0
							
			else:

				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						torch.cat([hidden_states_sys, hidden_states_vision_full, hidden_states_text], dim=1),
						torch.cat([hidden_states_sys, hidden_states_vision_full, hidden_states_text], dim=1),
						attention_masks,
						torch.cat([position_ids_sys, position_ids_vision_full, position_ids_vision_text], dim=1),
						torch.cat([position_ids_sys, position_ids_vision_full, position_ids_vision_text], dim=1),
						past_key_values,
						output_attentions,
						use_cache,
					)
				else:
					layer_outputs = decoder_layer(
						torch.cat([hidden_states_sys, hidden_states_vision_full, hidden_states_text], dim=1),
						torch.cat([hidden_states_sys, hidden_states_vision_full, hidden_states_text], dim=1),
						attention_masks,
						torch.cat([position_ids_sys, position_ids_vision_full, position_ids_vision_text], dim=1),
						torch.cat([position_ids_sys, position_ids_vision_full, position_ids_vision_text], dim=1),
						past_key_values,
						output_attentions,
						use_cache,
					)

				
				hidden_states_text = layer_outputs[0][:, vision_token_start_idx+image_token_newline_num:]
				hidden_states_sys = layer_outputs[0][:, :vision_token_start_idx]
				hidden_states_vision_full = layer_outputs[0][:, vision_token_start_idx:vision_token_start_idx+image_token_newline_num]

				if (i+1) in skip_layers:

					bs = hidden_states_vision_full.shape[0]
					image_features_full_with_newline = hidden_states_vision_full.clone()
					image_features_full_with_newline = image_features_full_with_newline.view(bs, image_token_len_per_side, image_token_len_per_side+1, -1)
					image_features_full = image_features_full_with_newline[:, :, :-1, :]
					image_features_full_newline = image_features_full_with_newline[:, :, -1:, :]
					reduce_factor = image_token_len_per_side // image_token_len_per_side_concise
					image_features_concise_newline = image_features_full_newline[:, ::reduce_factor, :, :].contiguous()

					image_features_concise = F.interpolate(
					image_features_full.permute(0, 3, 1, 2).contiguous().to(torch.float32),
						size=(image_token_len_per_side_concise, image_token_len_per_side_concise),
						mode='bilinear',
						align_corners=False
					).to(image_features_full.dtype)
					image_features_concise = image_features_concise.permute(0, 2, 3, 1).contiguous()
					image_features_concise_with_newline = torch.cat([image_features_concise, image_features_concise_newline], 2).flatten(1, 2)
					hidden_states_vision_concise = image_features_concise_with_newline
		

		hidden_states_text = self.norm(hidden_states_text)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states_text,)

		next_cache = None

		if not return_dict:
			return tuple(v for v in [hidden_states_text, next_cache, all_hidden_states, all_self_attns, aux_loss_total] if v is not None)
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states_text,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)


class CambrianQwenForCausalLM(Qwen2ForCausalLM, CambrianMetaForCausalLM):
	config_class = CambrianConfig

	def __init__(self, config):
		super(Qwen2ForCausalLM, self).__init__(config)

		self.model = CambrianQwenModel(config)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

		# Initialize weights and apply final processing
		self.post_init()

	def get_model(self):
		return self.model

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_masks: Optional[torch.Tensor] = None,
		attention_mask_c2f: Optional[torch.Tensor] = None,
		attention_masks_all2all: Optional[torch.Tensor] = None,
		image_valid_mask: Optional[torch.Tensor] = None,
		vision_full_attention_mask:Optional[torch.Tensor] = None,
		position_ids_sys: Optional[torch.LongTensor] = None,
		position_ids_vision_concise: Optional[torch.LongTensor] = None,
		position_ids_vision_full: Optional[torch.LongTensor] = None,
		position_ids_vision_text: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		images: Optional[torch.FloatTensor] = None,
		image_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
		image_sizes: Optional[List[List[int]]] = None,
		return_dict: Optional[bool] = None,
		gist_token_positions: Optional[torch.LongTensor] = None,
		cache_position = None
	) -> Union[Tuple, CausalLMOutputWithPast]:

		if inputs_embeds is None:
			(
				input_ids,
				inputs_embeds,
				inputs_embeds_vision_concise,
				vision_tower_aux_feature_list,
				vision_tower_aux_attention_masks_list,
				final_vision_feature_size,
				global_context_feature
			) = self.prepare_inputs_labels_for_multimodal(
				input_ids,
				images,
				image_aux_attention_masks_list,
				vision_full_attention_mask,
				image_sizes
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
			attention_masks=attention_masks,
			attention_mask_c2f=attention_mask_c2f,
			attention_masks_all2all=attention_masks_all2all,
			image_valid_mask=image_valid_mask,
			vision_full_attention_mask=vision_full_attention_mask,
			position_ids_sys=position_ids_sys,
			position_ids_vision_concise=position_ids_vision_concise,
			position_ids_vision_full=position_ids_vision_full,
			position_ids_vision_text=position_ids_vision_text,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			inputs_embeds_vision_concise=inputs_embeds_vision_concise,
			vision_tower_aux_feature_list=vision_tower_aux_feature_list,
			vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list, 
			final_vision_feature_size=final_vision_feature_size,
			global_context_feature=global_context_feature,
			gist_token_positions=gist_token_positions
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
		if labels is not None:
			labels = labels[:, -hidden_states.shape[1]:]
		logits = self.lm_head(hidden_states)
		logits = logits.float()

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
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
	


from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention, Qwen2DecoderLayer, Qwen2RMSNorm, rotate_half, repeat_kv
def apply_rotary_pos_emb(q, k, cos, sin, position_ids_q, position_ids_k, unsqueeze_dim=1):
	cos_q = cos[position_ids_q].unsqueeze(unsqueeze_dim)
	sin_q = sin[position_ids_q].unsqueeze(unsqueeze_dim)
	q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
	cos_k = cos[position_ids_k].unsqueeze(unsqueeze_dim)
	sin_k = sin[position_ids_k].unsqueeze(unsqueeze_dim)
	k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
	return q_embed, k_embed


# Adapted from Qwen2Attention.forward
def Qwen2SdpaAttention_forward(
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

	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

	attn_output = self.o_proj(attn_output)

	return attn_output, None, past_key_value

Qwen2SdpaAttention.forward = Qwen2SdpaAttention_forward

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

Qwen2DecoderLayer.forward = decoder_forward

AutoConfig.register("cambrian_qwen", CambrianConfig)
AutoModelForCausalLM.register(CambrianConfig, CambrianQwenForCausalLM)
