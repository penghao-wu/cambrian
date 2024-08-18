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
class CambrianConfig(LlamaConfig):
	model_type = "cambrian_llama"

	debug = "debug"


class CambrianLlamaModel(CambrianMetaModel, LlamaModel):
	config_class = CambrianConfig

	def __init__(self, config: LlamaConfig):
		super(CambrianLlamaModel, self).__init__(config)

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_masks: Optional[torch.Tensor] = None,
		attention_mask_c2f: Optional[torch.Tensor] = None,
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

		hidden_states = inputs_embeds

		for i, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states_text,)

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


			# if self.gradient_checkpointing and self.training:
			# 	layer_outputs = self._gradient_checkpointing_func(
			# 		decoder_layer.__call__,
			# 		torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_text], dim=1),
			# 		torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_text], dim=1),
			# 		attention_masks,
			# 		torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_text], dim=1),
			# 		torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_text], dim=1),
			# 		past_key_values,
			# 		output_attentions,
			# 		use_cache,
			# 	)
			# else:
			# 	layer_outputs = decoder_layer(
			# 		torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_text], dim=1),
			# 		torch.cat([hidden_states_sys, hidden_states_vision_concise, hidden_states_text], dim=1),
			# 		attention_masks,
			# 		torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_text], dim=1),
			# 		torch.cat([position_ids_sys, position_ids_vision_concise, position_ids_vision_text], dim=1),
			# 		past_key_values,
			# 		output_attentions,
			# 		use_cache,
			# 	)


			hidden_states_vision_concise_delta = layer_outputs[0][:, vision_token_start_idx:vision_token_start_idx+image_token_concise_newline_num] - hidden_states_vision_concise
			hidden_states_vision_concise = layer_outputs[0][:, vision_token_start_idx:vision_token_start_idx+image_token_concise_newline_num]
			hidden_states_text = layer_outputs[0][:, vision_token_start_idx+image_token_concise_newline_num:]
			hidden_states_sys = layer_outputs[0][:, :vision_token_start_idx]


			# hidden_states_vision_concise_1 = layer_outputs_1[0][:, vision_token_start_idx:vision_token_start_idx+image_token_concise_newline_num]
			# hidden_states_text_1 = layer_outputs_1[0][:, vision_token_start_idx+image_token_concise_newline_num:]
			# hidden_states_sys_1 = layer_outputs_1[0][:, :vision_token_start_idx]

			# hidden_states_vision_full = hidden_states_vision_concise
			# update vision full with concise
			hidden_states_vision_full = self.vision_sampler_layers[i](hidden_states_vision_full, hidden_states_vision_concise_delta, image_token_len_per_side, image_token_len_per_side_concise)

		hidden_states_text = self.norm(hidden_states_text)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states_text,)

		next_cache = None

		if not return_dict:
			return tuple(v for v in [hidden_states_text, next_cache, all_hidden_states, all_self_attns] if v is not None)
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states_text,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
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
		attention_masks: Optional[torch.Tensor] = None,
		attention_mask_c2f: Optional[torch.Tensor] = None,
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

AutoConfig.register("cambrian_llama", CambrianConfig)
AutoModelForCausalLM.register(CambrianConfig, CambrianLlamaForCausalLM)
