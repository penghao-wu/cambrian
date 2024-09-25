# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import re
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import time

import numpy as np
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import transformers
import tokenizers

import cambrian

from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from cambrian.train.cambrian_trainer import CambrianTrainer

from cambrian import conversation as conversation_lib

from cambrian.utils import IS_XLA_AVAILABLE, process_video_with_decord
from cambrian.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3, process_anyres_image
from cambrian.train.wandb_nan_alert_callback import NanInfAlertWandbCallback
from cambrian.model import CambrianLlamaForCausalLM, CambrianMistralForCausalLM, CambrianQwenForCausalLM
from cambrian.model.language_model.cambrian_phi3 import CambrianPhi3ForCausalLM
from PIL import Image

from ezcolorlog import root_logger as logger

from packaging import version


logger.setLevel(logging.WARNING)





local_rank = None

XLA_DISABLE_FUNCTIONALIZATION = bool(os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))

PRINT_LOGS = True


def print_rank0(*args):
	if local_rank in (0, -1) and PRINT_LOGS:
		print(*args)


def log_rank0(log):
	if local_rank in (0, -1) and PRINT_LOGS:
		logger.info(log, stacklevel=2)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
	version: Optional[str] = field(default="v0")
	freeze_backbone: bool = field(default=False)
	tune_mm_mlp_adapter: bool = field(default=False)
	vision_tower: Optional[str] = field(default=None)
	vision_tower_aux_list: Optional[str] = field(default=None)
	mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
	pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
	mm_projector_type: Optional[str] = field(default='linear')
	mm_use_im_start_end: bool = field(default=False)
	mm_use_im_patch_token: bool = field(default=True)
	mm_patch_merge_type: Optional[str] = field(default='flat')
	mm_vision_select_feature: Optional[str] = field(default="patch")
	vision_tower_aux_token_len_list: Optional[str] = field(default=None)
	num_query_group: Optional[int] = field(default=1)
	query_num_list: Optional[str] = field(default='[576]')
	connector_depth: Optional[int] = field(default=1)
	vision_hidden_size: Optional[int] = field(default=1024)
	connector_only: bool = field(default=True)

	num_of_vision_sampler_layers: Optional[int] = field(default=10)
	start_of_vision_sampler_layers: Optional[int] = field(default=16)
	stride_of_vision_sampler_layers: Optional[int] = field(default=1)

	# compressV
	max_num_image_crops: Optional[int] = field(default=1)
	per_crop_token_len: Optional[int] = field(default=576)
	compress_reduce_factor: Optional[int] = field(default=4)

	compress_v: Optional[bool] = field(default=False)
	compress_v_start_layer: Optional[int] = field(default=0)


@dataclass
class DataArguments:
	data_path: str = field(default=None,
						   metadata={"help": "Path to the training data."})
	lazy_preprocess: bool = False
	image_folder: Optional[str] = field(default=None)
	is_multimodal: bool = False
	image_aspect_ratio: str = 'square'
	image_position: int = 35  # depends on v1 conv


@dataclass
class TrainingArguments(transformers.TrainingArguments):
	cache_dir: Optional[str] = field(default=None)
	optim: str = field(default="adamw_torch")
	remove_unused_columns: bool = field(default=False)
	freeze_mm_mlp_adapter: bool = field(default=False)
	unfreeze_mm_vision_tower: bool = field(default=False)
	mpt_attn_impl: Optional[str] = field(default="triton")
	model_max_length: int = field(
		default=512,
		metadata={
			"help":
			"Maximum sequence length. Sequences will be right padded (and possibly truncated)."
		},
	)
	double_quant: bool = field(
		default=True,
		metadata={"help": "Compress the quantization statistics through double quantization."}
	)
	quant_type: str = field(
		default="nf4",
		metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
	)
	bits: int = field(
		default=16,
		metadata={"help": "How many bits to use."}
	)
	lora_enable: bool = False
	lora_r: int = 64
	lora_alpha: int = 16
	lora_dropout: float = 0.05
	lora_weight_path: str = ""
	lora_bias: str = "none"
	mm_projector_lr: Optional[float] = None
	mm_vision_sampler_lr: Optional[float] = None
	group_by_modality_length: bool = field(default=False)
	mm_vision_tower_lr: Optional[float] = None
	mm_vision_mlp_lr: Optional[float] = None

	# sanity check arg
	batch_size: Optional[int] = field(
		default=None,
		metadata={"help": "The total batch size for training. If passed, will be used to check that the "
						  "`per_device_train_batch_size` is set correctly."}
	)

	# GCSFS
	gcp_project: Optional[str] = field(default=None)
	"""Can also set GCP_PROJECT environment variable."""
	gcs_output_dir: Optional[str] = field(default=None)
	"""gs://<bucket>/<prefix>"""

	train_continue: bool = False
	resume_from_checkpoint: Optional[str] = ""

def maybe_zero_3(param, ignore_status=False, name=None):
	from deepspeed import zero
	from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
	if hasattr(param, "ds_id"):
		if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
			if not ignore_status:
				logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
		with zero.GatheredParameters([param]):
			param = param.data.detach().cpu().clone()
	else:
		param = param.detach().cpu().clone()
	return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
	if bias == "none":
		to_return = {k: t for k, t in named_params if "lora_" in k}
	elif bias == "all":
		to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
	elif bias == "lora_only":
		to_return = {}
		maybe_lora_bias = {}
		lora_bias_names = set()
		for k, t in named_params:
			if "lora_" in k:
				to_return[k] = t
				bias_name = k.split("lora_")[0] + "bias"
				lora_bias_names.add(bias_name)
			elif "bias" in k:
				maybe_lora_bias[k] = t
		for k, t in maybe_lora_bias:
			if bias_name in lora_bias_names:
				to_return[bias_name] = t
	else:
		raise NotImplementedError
	to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
	return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
	to_return = {k: t for k, t in named_params if "lora_" not in k}
	if require_grad_only:
		to_return = {k: t for k, t in to_return.items() if t.requires_grad}
	to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
	return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
	to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
	to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
	return to_return


def find_all_linear_names(model):
	cls = torch.nn.Linear
	lora_module_names = set()
	multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_tower_aux', 'vision_resampler', 'vision_sampler']
	for name, module in model.named_modules():
		if any(mm_keyword in name for mm_keyword in multimodal_keywords):
			continue
		if isinstance(module, cls):
			names = name.split('.')
			lora_module_names.add(names[0] if len(names) == 1 else names[-1])

	if 'lm_head' in lora_module_names: # needed for 16-bit
		lora_module_names.remove('lm_head')
	return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
								   output_dir: str):
	"""Collects the state dict and dump to disk."""
	output_dir = os.path.join('checkpoints', output_dir.split(os.sep)[-1])
	if getattr(trainer.args, "tune_mm_mlp_adapter", False):
		# Only save Adapter
		keys_to_match = ['mm_projector', 'pos_emb', 'vision_sampler', 'vision_sampler_layers', 'vision_query', 'image_newline', 'vision_mlp_layers']
		if getattr(trainer.args, "use_im_start_end", False):
			keys_to_match.extend(['embed_tokens', 'embed_in'])

		weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
		if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
			trainer.model.config.save_pretrained(output_dir)

		if not IS_XLA_AVAILABLE:
			raise NotImplementedError("Only XLA is supported for now.")

		import torch_xla.core.xla_model as xm
		ckpt_prefix = os.path.join(output_dir, "mm_projector")
		
		os.makedirs(output_dir, exist_ok=True)
		rank = xm.get_ordinal()
		world_size = xm.xrt_world_size()
		ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
		ckpt = {
			'model': weight_to_save,
			'shard_metadata': trainer.model.get_shard_metadata()
		}
		os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
		xm.save(ckpt, ckpt_path, master_only=False)
		print(f'checkpoint saved to {ckpt_path}\n', end='')
		return

	if trainer.deepspeed:
		torch.cuda.synchronize()
		trainer.save_model(output_dir)
		return

	trainer._save(output_dir)
   
def smart_tokenizer_and_embedding_resize(
	special_tokens_dict: Dict,
	tokenizer: transformers.PreTrainedTokenizer,
	model: transformers.PreTrainedModel,
):
	"""Resize tokenizer and embedding.

	Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
	"""
	num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
	model.resize_token_embeddings(len(tokenizer))

	if num_new_tokens > 0:
		input_embeddings = model.get_input_embeddings().weight.data
		output_embeddings = model.get_output_embeddings().weight.data

		input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
			dim=0, keepdim=True)
		output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
			dim=0, keepdim=True)

		input_embeddings[-num_new_tokens:] = input_embeddings_avg
		output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
				 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
	"""Tokenize a list of strings."""
	tokenized_list = [
		tokenizer(
			text,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		) for text in strings
	]
	input_ids = labels = [
		tokenized.input_ids[0] for tokenized in tokenized_list
	]
	input_ids_lens = labels_lens = [
		tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
		for tokenized in tokenized_list
	]
	return dict(
		input_ids=input_ids,
		labels=labels,
		input_ids_lens=input_ids_lens,
		labels_lens=labels_lens,
	)


def _mask_targets(target, tokenized_lens, speakers):
	# cur_idx = 0
	cur_idx = tokenized_lens[0]
	tokenized_lens = tokenized_lens[1:]
	target[:cur_idx] = IGNORE_INDEX
	for tokenized_len, speaker in zip(tokenized_lens, speakers):
		if speaker == "human":
			target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
		cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
	"""Add speaker and start/end signal on each round."""
	BEGIN_SIGNAL = "### "
	END_SIGNAL = "\n"
	conversation = header
	for sentence in source:
		from_str = sentence["from"]
		if from_str.lower() == "human":
			from_str = conversation_lib.default_conversation.roles[0]
		elif from_str.lower() == "gpt":
			from_str = conversation_lib.default_conversation.roles[1]
		else:
			from_str = 'unknown'
		sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
							 sentence["value"] + END_SIGNAL)
		if get_conversation:
			conversation += sentence["value"]
	conversation += BEGIN_SIGNAL
	return conversation


def preprocess_multimodal(
	sources: Sequence[str],
	data_args: DataArguments
) -> Dict:
	is_multimodal = data_args.is_multimodal
	if not is_multimodal:
		return sources

	for source in sources:
		for sentence in source:
			if DEFAULT_IMAGE_TOKEN in sentence['value']:
				sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
				sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
				sentence['value'] = sentence['value'].strip()
				if "mmtag" in conversation_lib.default_conversation.version:
					sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
			replace_token = DEFAULT_IMAGE_TOKEN
			if data_args.mm_use_im_start_end:
				replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
			sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

	return sources

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
	# roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
	roles = {"human": "user", "gpt": "assistant"}

	# Add image tokens to tokenizer as a special tokens
	# Use a deepcopy of tokenizer so that we don't modify on the tokenizer
	tokenizer = copy.deepcopy(tokenizer)
	# When there is actually an image, we add the image tokens as a special token
	if has_image:
		tokenizer.add_tokens(["<image>"], special_tokens=True)

	image_token_index = tokenizer.convert_tokens_to_ids("<image>")
	im_start, im_end = tokenizer.additional_special_tokens_ids
	# unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
	unmask_tokens_idx =  [198, im_start, im_end]
	nl_tokens = tokenizer("\n").input_ids

	# Reset Qwen chat templates so that it won't include system message every time we apply
	chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
	tokenizer.chat_template = chat_template

	# _system = tokenizer("system").input_ids + nl_tokens
	# _user = tokenizer("user").input_ids + nl_tokens
	# _assistant = tokenizer("assistant").input_ids + nl_tokens

	# Apply prompt templates
	input_ids, targets = [], []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != roles["human"]:
			source = source[1:]

		input_id, target = [], []

		# New version, use apply chat template
		# Build system message for each sentence
		input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
		target += [IGNORE_INDEX] * len(input_id)

		for conv in source:
			# Make sure llava data can load
			try:
				role = conv["role"]
				content = conv["content"]
			except:
				role = conv["from"]
				content = conv["value"]

			role =  roles.get(role, role)
			
			conv = [{"role" : role, "content" : content}]
			encode_id = tokenizer.apply_chat_template(conv)
			input_id += encode_id
			if role in ["user", "system"]:
				target += [IGNORE_INDEX] * len(encode_id)
			else:
				target += encode_id
		

					
		assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
		for idx, encode_id in enumerate(input_id):
			if encode_id in unmask_tokens_idx:
				target[idx] = encode_id
			if encode_id == image_token_index:
				input_id[idx] = IMAGE_TOKEN_INDEX
		input_ids.append(input_id)
		targets.append(target)
	input_ids = torch.tensor(input_ids, dtype=torch.long)
	targets = torch.tensor(targets, dtype=torch.long)

	return dict(
		input_ids=input_ids,  # tensor(bs x seq_len)
		labels=targets,  # tensor(bs x seq_len)
	)

def preprocess_llama_3(
	sources,
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False
) -> Dict:
	conv = conversation_lib.default_conversation.copy()
	roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

	conversations = []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != conv.roles[0]:
			# Skip the first one if it is not from human
			source = source[1:]

		conv.messages = []
		for j, sentence in enumerate(source):
			role = roles[sentence["from"]]
			assert role == conv.roles[j % 2], f"{i}"
			conv.append_message(role, sentence["value"])
		prompt = conv.get_prompt()
		if prompt.endswith("<|start_header_id|>assistant<|end_header_id|>"):
			prompt = prompt[:-len("<|start_header_id|>assistant<|end_header_id|>")]
		conversations.append(prompt)

	# Tokenize conversations

	if has_image:
		input_ids = torch.stack([tokenizer_image_token_llama3(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
	else:
		input_ids = tokenizer(
			conversations,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		).input_ids

	targets = input_ids.clone()

	assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

	# Mask targets
	sep = "<|eot_id|>"
	for conversation, target in zip(conversations, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split("<|eot_id|>")
		
		cur_len = 0

		for i, rou in enumerate(rounds):
			if rou == "":
				break

			rou += sep
			
			# System Prompt
			if i == 0:
				round_len = len(tokenizer(rou).input_ids)
				# Don't predict system prompt
				target[cur_len : cur_len + round_len] = IGNORE_INDEX
				cur_len += round_len
			# User Prompt
			elif i % 2 == 1:
				if i==1 and has_image:
					round_len = len(tokenizer_image_token_llama3(rou, tokenizer))
				else:
					round_len = len(tokenizer(rou).input_ids)
				# Don't predict system prompt
				target[cur_len : cur_len + round_len] = IGNORE_INDEX
				cur_len += round_len
			# Model Reponse
			elif i % 2 == 0:
				round_len = len(tokenizer(rou).input_ids)
				# Don't predict system prompt
				target[cur_len : cur_len + 3] = IGNORE_INDEX
				cur_len += round_len

			
		target[cur_len:] = IGNORE_INDEX

		# if cur_len < tokenizer.model_max_length:
		#     if cur_len != total_len:
		#         target[:] = IGNORE_INDEX
				# print(
				#     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
				#     f" (ignored)"
				# )
		
	return dict(
		input_ids=input_ids,
		labels=targets,
	)

def preprocess_llama_2(
	sources,
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False
) -> Dict:
	conv = conversation_lib.default_conversation.copy()
	roles = {"human": conv.roles[0], "gpt": conv.roles[1]}


	# Apply prompt templates
	conversations = []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != conv.roles[0]:
			# Skip the first one if it is not from human
			source = source[1:]

		conv.messages = []
		for j, sentence in enumerate(source):
			role = roles[sentence["from"]]
			assert role == conv.roles[j % 2], f"{i}"
			conv.append_message(role, sentence["value"])
		conversations.append(conv.get_prompt())

	# Tokenize conversations

	if has_image:
		input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
	else:
		input_ids = tokenizer(
			conversations,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		).input_ids

	targets = input_ids.clone()

	assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

	# Mask targets
	sep = "[/INST] "
	for conversation, target in zip(conversations, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split(conv.sep2)
		cur_len = 1
		target[:cur_len] = IGNORE_INDEX
		for i, rou in enumerate(rounds):
			if rou == "":
				break

			parts = rou.split(sep)
			if len(parts) != 2:
				break
			parts[0] += sep

			if has_image:
				round_len = len(tokenizer_image_token(rou, tokenizer))
				instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
			else:
				round_len = len(tokenizer(rou).input_ids)
				instruction_len = len(tokenizer(parts[0]).input_ids) - 2

			target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

			cur_len += round_len
		target[cur_len:] = IGNORE_INDEX

		if cur_len < tokenizer.model_max_length:
			if cur_len != total_len:
				target[:] = IGNORE_INDEX
				print_rank0(
					f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}, conversation is {conversation}."
					f" (ignored)"
				)

	return dict(
		input_ids=input_ids,
		labels=targets,
	)


def preprocess_v1(
	sources,
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False
) -> Dict:
	conv = conversation_lib.default_conversation.copy()
	roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

	# Apply prompt templates
	conversations = []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != conv.roles[0]:
			# Skip the first one if it is not from human
			source = source[1:]

		conv.messages = []
		for j, sentence in enumerate(source):
			role = roles[sentence["from"]]
			assert role == conv.roles[j % 2], f"{i}"
			conv.append_message(role, sentence["value"])
		conversations.append(conv.get_prompt())

	# Tokenize conversations

	if has_image:
		input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
	else:
		input_ids = tokenizer(
			conversations,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		).input_ids

	targets = input_ids.clone()

	assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

	# Mask targets
	sep = conv.sep + conv.roles[1] + ": "
	for conversation, target in zip(conversations, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split(conv.sep2)
		cur_len = 1
		target[:cur_len] = IGNORE_INDEX
		for i, rou in enumerate(rounds):
			if rou == "":
				break

			parts = rou.split(sep)
			if len(parts) != 2:
				break
			parts[0] += sep

			if has_image:
				round_len = len(tokenizer_image_token(rou, tokenizer))
				instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
			else:
				round_len = len(tokenizer(rou).input_ids)
				instruction_len = len(tokenizer(parts[0]).input_ids) - 2

			if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
				round_len -= 1
				instruction_len -= 1

			target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

			cur_len += round_len
		target[cur_len:] = IGNORE_INDEX

		if cur_len < tokenizer.model_max_length:
			if cur_len != total_len:
				target[:] = IGNORE_INDEX
				print_rank0(
					f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
					f" (ignored)"
				)

	return dict(
		input_ids=input_ids,
		labels=targets,
	)


def preprocess_mpt(
	sources,
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False
) -> Dict:
	conv = conversation_lib.default_conversation.copy()
	roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

	# Apply prompt templates
	conversations = []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != conv.roles[0]:
			# Skip the first one if it is not from human
			source = source[1:]

		conv.messages = []
		for j, sentence in enumerate(source):
			role = roles[sentence["from"]]
			assert role == conv.roles[j % 2], f"{i}"
			conv.append_message(role, sentence["value"])
		conversations.append(conv.get_prompt())

	# Tokenize conversations

	if has_image:
		input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
	else:
		input_ids = tokenizer(
			conversations,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		).input_ids

	targets = input_ids.clone()
	assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

	# Mask targets
	sep = conv.sep + conv.roles[1]
	for conversation, target in zip(conversations, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split(conv.sep)
		re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
		for conv_idx in range(3, len(rounds), 2):
			re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
		cur_len = 1
		target[:cur_len] = IGNORE_INDEX
		for i, rou in enumerate(re_rounds):
			if rou == "":
				break

			parts = rou.split(sep)
			if len(parts) != 2:
				break
			parts[0] += sep

			if has_image:
				round_len = len(tokenizer_image_token(rou, tokenizer))
				instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
			else:
				round_len = len(tokenizer(rou).input_ids)
				instruction_len = len(tokenizer(parts[0]).input_ids) - 1

			if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
				round_len += 1
				instruction_len += 1

			target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

			cur_len += round_len
		target[cur_len:] = IGNORE_INDEX

		if cur_len < tokenizer.model_max_length:
			if cur_len != total_len:
				target[:] = IGNORE_INDEX
				print_rank0(
					f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
					f" (ignored)"
				)

	return dict(
		input_ids=input_ids,
		labels=targets,
	)


def preprocess_plain(
	sources: Sequence[str],
	tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
	# add end signal and concatenate together
	conversations = []
	for source in sources:
		assert len(source) == 2
		assert DEFAULT_IMAGE_TOKEN in source[0]['value']
		source[0]['value'] = DEFAULT_IMAGE_TOKEN
		conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
		conversations.append(conversation)
	# tokenize conversations
	input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
	targets = copy.deepcopy(input_ids)
	for target, source in zip(targets, sources):
		tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
		target[:tokenized_len] = IGNORE_INDEX

	return dict(input_ids=input_ids, labels=targets)


def preprocess_phi3(
	sources,
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False
) -> Dict:
	conv = conversation_lib.default_conversation.copy()
	roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

	# Apply prompt templates
	conversations = []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != conv.roles[0]:
			# Skip the first one if it is not from human
			source = source[1:]

		conv.messages = []
		for j, sentence in enumerate(source):
			role = roles[sentence["from"]]
			assert role == conv.roles[j % 2], f"{i}"
			conv.append_message(role, sentence["value"])
		conversations.append(conv.get_prompt())

	# Tokenize conversations

	if has_image:
		input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
	else:
		input_ids = tokenizer(
			conversations,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		).input_ids

	targets = input_ids.clone()

	assert conv.sep_style == conversation_lib.SeparatorStyle.PHI3

	# Mask targets
	sep = conv.sep + conv.roles[1]
	for conversation, target in zip(conversations, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split(conv.sep)
		re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
		for conv_idx in range(3, len(rounds), 2):
			re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
		cur_len = 1
		target[:cur_len] = IGNORE_INDEX
		
		for i, rou in enumerate(re_rounds):
			if rou == "":
				break

			parts = rou.split(sep)
			if len(parts) != 2:
				break
			parts[0] += sep
			if has_image:
				round_len = len(tokenizer_image_token(rou, tokenizer))
				instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
			else:
				round_len = len(tokenizer(rou).input_ids)
				instruction_len = len(tokenizer(parts[0]).input_ids) - 1
			if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
				round_len -= 1
				instruction_len -= 1
			if i != 0: # remove the first \n token
				round_len -= 1
				instruction_len -= 1

			target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

			cur_len += round_len
		target[cur_len:] = IGNORE_INDEX

		if cur_len < tokenizer.model_max_length:
			if cur_len != total_len:
				target[:] = IGNORE_INDEX
				print_rank0(
					f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}, conversation is {conversation}."
					f" (ignored)"
				)

	return dict(
		input_ids=input_ids,
		labels=targets,
	)


def preprocess(
	sources: Sequence[str],
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False
) -> Dict:

	"""
	Given a list of sources, each is a conversation list. This transform:
	1. Add signal '### ' at the beginning each sentence, with end signal '\n';
	2. Concatenate conversations together;
	3. Tokenize the concatenated conversation;
	4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
	"""
	if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
		return preprocess_plain(sources, tokenizer)
	if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
		return preprocess_llama_2(sources, tokenizer, has_image=has_image)
	if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
		return preprocess_llama_3(sources, tokenizer, has_image=has_image)
	if conversation_lib.default_conversation.version.startswith("v1"):
		return preprocess_v1(sources, tokenizer, has_image=has_image)
	if conversation_lib.default_conversation.version == "mpt":
		return preprocess_mpt(sources, tokenizer, has_image=has_image)
	if conversation_lib.default_conversation.version == "qwen":
		return preprocess_qwen(sources, tokenizer, has_image=has_image)
	if conversation_lib.default_conversation.version == "phi3":
		return preprocess_phi3(sources, tokenizer, has_image=has_image)
	
	# add end signal and concatenate together
	conversations = []
	for source in sources:
		header = f"{conversation_lib.default_conversation.system}\n\n"
		conversation = _add_speaker_and_signal(header, source)
		conversations.append(conversation)
	# tokenize conversations
	def get_tokenize_len(prompts):
		return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

	if has_image:
		input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
	else:
		conversations_tokenized = _tokenize_fn(conversations, tokenizer)
		input_ids = conversations_tokenized["input_ids"]

	targets = copy.deepcopy(input_ids)
	for target, source in zip(targets, sources):
		if has_image:
			tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
		else:
			tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
		speakers = [sentence["from"] for sentence in source]
		_mask_targets(target, tokenized_lens, speakers)

	return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
	"""Dataset for supervised fine-tuning."""

	def __init__(self, data_path: str,
				 tokenizer: transformers.PreTrainedTokenizer,
				 data_args: DataArguments):
		super(LazySupervisedDataset, self).__init__()

		self.tokenizer = tokenizer
		self.data_path = data_path
		self.data_args = data_args
		self.length = self._get_length()

	def _get_length(self):
		"""Calculates the number of samples in the .jsonl file."""
		with open(self.data_path, 'r') as file:
			for i, _ in enumerate(file):
				pass
		return i + 1

	def __len__(self):
		"""Returns the number of samples in the dataset."""
		return self.length


	def _compute_lengths(self):
		"""Compute and cache lengths of conversations in the dataset."""
		if hasattr(self, 'length_list') and hasattr(self, 'modality_length_list'):
			# Return cached values if already computed
			return self.length_list, self.modality_length_list

		self.length_list = []
		self.modality_length_list = []
		with open(self.data_path, 'r') as file:
			for line in file:
				sample = json.loads(line.strip())
				img_tokens = 128 if self._has_image(sample) else 0
				cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
				self.length_list.append(cur_len + img_tokens)
				modality_len = cur_len if 'image' in sample else -cur_len
				self.modality_length_list.append(modality_len)
		return self.length_list, self.modality_length_list

	@property
	def lengths(self):
		length_list, _ = self._compute_lengths()
		return length_list

	@property
	def modality_lengths(self):
		_, modality_length_list = self._compute_lengths()
		return modality_length_list

	def _has_image(self, sample: dict) -> bool:
		return "image" in sample and not str(sample['image']) in ['', 'None', 'none', 'nan']
	
	def process_image(self, image_file, overwrite_image_aspect_ratio=None):
		image_folder = self.data_args.image_folder
		processor_aux_list = self.data_args.image_processor_aux_list
		processor = processor_aux_list[0]
		# print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
		try:
			image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
		except Exception as exn:
			print(f"Failed to open image {image_file}. Exception:", exn)
			raise exn
		image_size = image.size
		image_aspect_ratio = self.data_args.image_aspect_ratio
		if overwrite_image_aspect_ratio is not None:
			image_aspect_ratio = overwrite_image_aspect_ratio
		if image_aspect_ratio not in ['pad', 'anyres']:
			raise NotImplementedError("Only pad and anyres is supported for now.")

		image2crops_num = 1
		if image_aspect_ratio == "anyres":
			image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
			image2crops_num = image.shape[0]
		elif image_aspect_ratio == "pad":
			def expand2square(pil_img, background_color):
				width, height = pil_img.size
				if width == height:
					return pil_img
				elif width > height:
					result = Image.new(pil_img.mode, (width, width), background_color)
					result.paste(pil_img, (0, (width - height) // 2))
					return result
				else:
					result = Image.new(pil_img.mode, (height, height), background_color)
					result.paste(pil_img, ((height - width) // 2, 0))
					return result
			image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
			image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
			image = image.unsqueeze(0)
		return image, image_size, image2crops_num
	
	def __getitem__(self, i) -> Dict[str, torch.Tensor]:
		# TODO: define number of retries somewhere else
		num_base_retries = 3
		num_final_retries = 300

		# try the current sample first
		for attempt_idx in range(num_base_retries):
			try:
				sample = self._get_item(i)
				return sample
			except Exception as e:
				# sleep 1s in case it is a cloud disk issue
				print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
				time.sleep(1)

		# try other samples, in case it is file corruption issue
		for attempt_idx in range(num_base_retries):
			try:
				next_index = min(i + 1, len(self.lengths) - 1)
				# sample_idx = random.choice(range(len(self)))
				sample = self._get_item(next_index)
				return sample
			except Exception as e:
				# no need to sleep
				print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
				pass

		try:
			sample = self._get_item(i)
			return sample
		except Exception as e:
			raise e

	def _get_item(self, i) -> Dict[str, torch.Tensor]:
		#sources = self.list_data_dict[i]
		with open(self.data_path, 'r') as file:
			for idx, line in enumerate(file):
				if idx == i:
					sources = json.loads(line.strip())
					break
		sources = {"id":"000000398214","image":"coco/train2017/000000398214.jpg","conversations":[{"from":"human","value":"What is it\n<image>"},{"from":"gpt","value":"City"}]}
		dat = sources
		if isinstance(i, int):
			sources = [sources]
		assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
		has_image = self._has_image(dat)

		images = []
		# for a anyres image, we treat each crop as a single image and we need to repeat the <image>, so we record this value here
		image2crops_nums = []
		if has_image:
			image_file = dat['image']
			if type(image_file) is list:
				if len(image_file) > 1:
					image = [self.process_image(f, "pad") for f in image_file]
				else:
					image = [self.process_image(f) for f in image_file]
			else:
				image = [self.process_image(image_file)]

			images = torch.cat([img[0] for img in image])
			image2crops_nums = [img[2] for img in image]
			sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

		elif "video" in sources[0]:
			video_file = dat["video"]
			video_folder = self.data_args.video_folder
			video_file = os.path.join(video_folder, video_file)
			suffix = video_file.split(".")[-1]
			if not os.path.exists(video_file):
				print("File {} not exist!".format(video_file))

			try:
				if "shareVideoGPTV" in video_file:
					frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
					frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

					# TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
					if self.data_args.force_sample:
						num_frames_to_sample = self.data_args.frames_upbound
					else:
						num_frames_to_sample = 10

					avg_fps = 2
					
					total_frames = len(frame_files)
					sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


					frame_time = [i/2 for i in sampled_indices]
					frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

					video_time = total_frames / avg_fps

					# Read and store the sampled frames
					video = []
					for idx in sampled_indices:
						frame_path = frame_files[idx]
						try:
							with Image.open(frame_path) as img:
								frame = img.convert("RGB")
								video.append(frame)
						except IOError:
							print(f"Failed to read frame at path: {frame_path}")
				else:
					video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

				processor = self.data_args.image_processor
				image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
				if self.data_args.add_time_instruction:
					time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
					sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
				image = [(image, video[0].size, "video")]
				images = torch.cat([img[0] for img in image])
				image2crops_nums = [1] * images.shape[0]
				sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
				# print(sources)
			except Exception as e:
				print(f"Error: {e}")
				print(f"Failed to read video file: {video_file}")
				return self._get_item(i + 1)
			sources = preprocess_multimodal(
				copy.deepcopy([e["conversations"] for e in sources]),
				self.data_args)
		else:
			sources = copy.deepcopy([e["conversations"] for e in sources])

		has_image = has_image or "video" in sources[0]

		data_dict = preprocess(
			sources,
			self.tokenizer,
			has_image=has_image)
		if isinstance(i, int):
			data_dict = dict(input_ids=data_dict["input_ids"][0],
							 labels=data_dict["labels"][0])
		if (data_dict['labels']!=IGNORE_INDEX).sum()==0:
			return self._get_item(i+1)
		
		# check whether image tokens will be truncated, if so, discard this sample
		if has_image or self.data_args.is_multimodal:    
			max_num_image_crops = self.data_args.max_num_image_crops
			per_crop_token_len = self.data_args.per_crop_token_len
			num_images = (data_dict['input_ids'] == IMAGE_TOKEN_INDEX).sum()
			if num_images == 0:
				pre_text_token_num = 0
			else:
				last_image_token_index = torch.where(data_dict['input_ids'] == IMAGE_TOKEN_INDEX)[0].tolist()[-1]
				pre_text_token_num = (last_image_token_index+1) - num_images

			if pre_text_token_num + (max_num_image_crops*(per_crop_token_len+1)) >= self.tokenizer.model_max_length-1:
				print("Skip this sample as its image tokens padded get truncated")
				return self._get_item(i + 1)

		# image exist in the data
		if has_image:
			data_dict['images'] = images
		elif self.data_args.is_multimodal:
			# image does not exist in the data, but the model is multimodal
			processor = self.data_args.image_processor_aux_list[0]
			data_dict['images'] = torch.zeros(self.data_args.max_num_image_crops, 3, processor.crop_size['height'], processor.crop_size['width'])
		data_dict['image2crops_nums'] = image2crops_nums
		return data_dict

def prepare_image_information(per_crop_token_len, compress_reduce_factor, is_dummy=False, dummy_num=1):
	height = width = int(per_crop_token_len**0.5)
	height_compress = width_compress = height // compress_reduce_factor

	if is_dummy:
		attention_mask_image_full = torch.zeros((dummy_num*(height*width),), dtype=torch.bool)
		attention_mask_newline_full = torch.zeros((dummy_num,), dtype=torch.bool)
		position_ids_image_full = torch.zeros((dummy_num*(height*width),), dtype=torch.long)
		position_ids_newline_full = torch.zeros((dummy_num,), dtype=torch.long)

		attention_mask_image_compress = torch.zeros((dummy_num*height_compress*width_compress,), dtype=torch.bool)
		position_ids_image_compress = torch.zeros((dummy_num*height_compress*width_compress,), dtype=torch.long)
	else:
		attention_mask_image_full_withnewline = torch.ones((height * width+1,), dtype=torch.bool)
		position_ids_image_full_withnewline = (attention_mask_image_full_withnewline.cumsum(0)-1).to(torch.long)

		attention_mask_image_full = attention_mask_image_full_withnewline[:-1]
		attention_mask_newline_full = attention_mask_image_full_withnewline[-1:]
		position_ids_image_full = position_ids_image_full_withnewline[:-1]
		position_ids_newline_full = position_ids_image_full_withnewline[-1:]

		attention_mask_image_compress = torch.ones((height_compress * width_compress,), dtype=torch.bool)
		position_ids_image_compress = (attention_mask_image_compress.cumsum(0)-1).to(torch.long)

	
	image_info = {}
	image_info['attention_mask_image_full'] = attention_mask_image_full
	image_info['attention_mask_newline_full'] = attention_mask_newline_full
	image_info['attention_mask_image_compress'] = attention_mask_image_compress

	image_info['position_ids_image_full'] = position_ids_image_full
	image_info['position_ids_newline_full'] = position_ids_newline_full
	image_info['position_ids_image_compress'] = position_ids_image_compress
	
	return image_info

def calculate_causal_attention_mask(position_ids_q, position_ids_kv, attention_mask_kv, dtype=torch.bfloat16):
	min_dtype = torch.finfo(dtype).min
	bs = position_ids_q.shape[0]
	position_ids_q = position_ids_q.view(bs, -1, 1)
	position_ids_kv = position_ids_kv.view(bs, -1, 1)
	causal_mask = position_ids_q >= position_ids_kv.transpose(1, 2)
	causal_mask = causal_mask.to(dtype).view(bs, 1, position_ids_q.shape[1], position_ids_kv.shape[1])
	attention_mask_4d = attention_mask_kv.view(bs, 1, 1, -1).repeat(1, 1, position_ids_q.shape[1], 1)

	causal_mask = causal_mask.masked_fill(causal_mask == 0, min_dtype)
	causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)
	causal_mask = causal_mask.masked_fill(attention_mask_4d == 0, min_dtype)

	return causal_mask
	

def prepare_multimodal_data(input_ids, labels, attention_mask, max_num_image_crops, per_crop_token_len, compress_reduce_factor, max_length=2048, pad_token_id=0):

	input_ids_text = []

	attention_mask_image_full = []
	attention_mask_image_compress = []
	attention_mask_newline_full = []
	attention_mask_text = []

	position_ids_image_full = []
	position_ids_image_compress = []
	position_ids_newline_full = []
	position_ids_text = []

	# Note that the labels here will already be shifted by 1
	labels_image_full = []
	labels_newline_full = []
	labels_text = []

	for batch_idx, cur_input_ids in enumerate(input_ids):
		cur_attention_mask = attention_mask[batch_idx]
		cur_labels = labels[batch_idx]
		num_image_crops = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
		if num_image_crops == 0:
			input_ids_text.append(cur_input_ids)

			attention_mask_image_full.append(torch.zeros((max_num_image_crops*per_crop_token_len,), dtype=torch.bool))
			attention_mask_image_compress.append(torch.zeros((max_num_image_crops*(per_crop_token_len//compress_reduce_factor**2),), dtype=torch.bool))
			attention_mask_newline_full.append(torch.zeros((max_num_image_crops,), dtype=torch.bool))
			attention_mask_text.append(cur_attention_mask)

			position_ids_image_full.append(torch.zeros((max_num_image_crops*per_crop_token_len,), dtype=torch.long))
			position_ids_image_compress.append(torch.zeros((max_num_image_crops*(per_crop_token_len//compress_reduce_factor**2),), dtype=torch.long))
			position_ids_newline_full.append(torch.zeros((max_num_image_crops,), dtype=torch.long))
			position_ids_text.append((cur_attention_mask.cumsum(0)-1).to(torch.long))

			labels_image_full.append(torch.full((max_num_image_crops*per_crop_token_len,), IGNORE_INDEX, dtype=torch.long))
			labels_newline_full.append(torch.full((max_num_image_crops,), IGNORE_INDEX, dtype=torch.long))
			labels_text.append(cur_labels)

			continue

		cur_input_ids_text = []

		cur_attention_mask_image_full = []
		cur_attention_mask_image_compress = []
		cur_attention_mask_newline_full = []
		cur_attention_mask_text = []

		cur_position_ids_image_full = []
		cur_position_ids_image_compress = []
		cur_position_ids_newline_full = []
		cur_position_ids_text = []

		cur_labels_image_full = []
		cur_labels_newline_full = []
		cur_labels_text = []

		image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
		position_id_count = 0
		for i in range(len(image_token_indices) - 1):
			# prepare everything for the text part
			cur_len_text = image_token_indices[i+1] - (image_token_indices[i]+1)
			if cur_len_text > 0:
				cur_input_ids_text.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
				cur_attention_mask_text.append(cur_attention_mask[image_token_indices[i]+1:image_token_indices[i+1]])
				cur_position_ids_text.append(torch.arange(position_id_count, position_id_count+cur_len_text, dtype=torch.long))
				position_id_count += cur_len_text

				if cur_len_text == 1:
					# single text token between images or at the end
					cur_labels_text.append(torch.full((1, ), IGNORE_INDEX, dtype=torch.long))
				elif cur_len_text > 1:
					# multiple text tokens, shift the labels by 1
					cur_labels_text.append(torch.cat([cur_labels[image_token_indices[i]+1:image_token_indices[i+1]][1:], torch.full((1, ), IGNORE_INDEX, dtype=torch.long)]))

			# prepare everything for images and newlines
			# Here we do not consider unpadding thing or spatial concat and always append a newline after each image crop

			if i < len(image_token_indices) - 2:
				cur_image_info = prepare_image_information(per_crop_token_len, compress_reduce_factor, is_dummy=False)
				cur_attention_mask_image_full.append(cur_image_info['attention_mask_image_full'])
				cur_attention_mask_image_compress.append(cur_image_info['attention_mask_image_compress'])
				cur_attention_mask_newline_full.append(cur_image_info['attention_mask_newline_full'])

				cur_position_ids_image_full.append(cur_image_info['position_ids_image_full']+position_id_count)
				cur_position_ids_image_compress.append(cur_image_info['position_ids_image_compress']+position_id_count)
				cur_position_ids_newline_full.append(cur_image_info['position_ids_newline_full']+position_id_count)

				position_id_count += per_crop_token_len+1
				

				cur_labels_image_full.append(torch.full((per_crop_token_len,) , IGNORE_INDEX, dtype=torch.long))
				# most of the labels for newlines should be IGNORE but the very last one
				# next token is text
				if i == num_image_crops-1 or image_token_indices[i+1] +1 != image_token_indices[i+2]:
					cur_labels_newline_full.append(cur_labels[image_token_indices[i+1]+1:image_token_indices[i+1]+2])
				# next token is image
				else:
					cur_labels_newline_full.append(torch.full((1,), IGNORE_INDEX, dtype=torch.long))

		# add dummy images & newlines to max_num_image_crops
		if num_image_crops < max_num_image_crops:
			num_image_crops_padded = max_num_image_crops - num_image_crops

			cur_image_info_dummy = prepare_image_information(per_crop_token_len, compress_reduce_factor, is_dummy=True, dummy_num=num_image_crops_padded)

			cur_attention_mask_image_full.append(cur_image_info_dummy['attention_mask_image_full'])
			cur_attention_mask_image_compress.append(cur_image_info_dummy['attention_mask_image_compress'])
			cur_attention_mask_newline_full.append(cur_image_info_dummy['attention_mask_newline_full'])

			cur_position_ids_image_full.append(cur_image_info_dummy['position_ids_image_full'])
			cur_position_ids_image_compress.append(cur_image_info_dummy['position_ids_image_compress'])
			cur_position_ids_newline_full.append(cur_image_info_dummy['position_ids_newline_full'])

			cur_labels_image_full.append(torch.full((per_crop_token_len*num_image_crops_padded,) , IGNORE_INDEX, dtype=torch.long))

			cur_labels_newline_full.append(torch.full((num_image_crops_padded,), IGNORE_INDEX, dtype=torch.long))

		input_ids_text.append(torch.cat(cur_input_ids_text))

		attention_mask_image_full.append(torch.cat(cur_attention_mask_image_full))
		attention_mask_image_compress.append(torch.cat(cur_attention_mask_image_compress))
		attention_mask_newline_full.append(torch.cat(cur_attention_mask_newline_full))
		attention_mask_text.append(torch.cat(cur_attention_mask_text))

		position_ids_image_full.append(torch.cat(cur_position_ids_image_full))
		position_ids_image_compress.append(torch.cat(cur_position_ids_image_compress))
		position_ids_newline_full.append(torch.cat(cur_position_ids_newline_full))
		position_ids_text.append(torch.cat(cur_position_ids_text))

		labels_image_full.append(torch.cat(cur_labels_image_full))
		labels_newline_full.append(torch.cat(cur_labels_newline_full))
		labels_text.append(torch.cat(cur_labels_text))


	# truncate each text and right pad them to the fixed max length
	non_text_len = max_num_image_crops*(per_crop_token_len+1)
	max_text_len = max_length - non_text_len
	for batch_idx in range(len(input_ids)):
		cur_text_len = len(input_ids_text[batch_idx])
		if cur_text_len > max_text_len:
			input_ids_text[batch_idx] = input_ids_text[batch_idx][:max_text_len]
			attention_mask_text[batch_idx] = attention_mask_text[batch_idx][:max_text_len]
			position_ids_text[batch_idx] = position_ids_text[batch_idx][:max_text_len]
			labels_text[batch_idx] = labels_text[batch_idx][:max_text_len]
		elif cur_text_len < max_text_len:
			pad_len = max_text_len - cur_text_len
			input_ids_text[batch_idx] = torch.cat([input_ids_text[batch_idx], torch.full((pad_len, ), pad_token_id, dtype=input_ids_text[batch_idx].dtype)])
			attention_mask_text[batch_idx] = torch.cat([attention_mask_text[batch_idx], torch.full((pad_len, ), 0, dtype=torch.bool)])
			position_ids_text[batch_idx] = torch.cat([position_ids_text[batch_idx], torch.full((pad_len, ), 0, dtype=torch.long)])
			labels_text[batch_idx] = torch.cat([labels_text[batch_idx], torch.full((pad_len, ), IGNORE_INDEX, dtype=torch.long)])

	input_ids_text = torch.stack(input_ids_text)
	attention_mask_image_full = torch.stack(attention_mask_image_full)
	attention_mask_image_compress = torch.stack(attention_mask_image_compress)
	attention_mask_newline_full = torch.stack(attention_mask_newline_full)
	attention_mask_text = torch.stack(attention_mask_text)

	position_ids_image_full = torch.stack(position_ids_image_full)
	position_ids_image_compress = torch.stack(position_ids_image_compress)
	position_ids_newline_full = torch.stack(position_ids_newline_full)
	position_ids_text = torch.stack(position_ids_text)

	labels_image_full = torch.stack(labels_image_full)
	labels_newline_full = torch.stack(labels_newline_full)
	labels_text = torch.stack(labels_text)

	labels = torch.cat([labels_image_full, labels_newline_full, labels_text], 1)
	attention_mask = torch.cat([attention_mask_image_full, attention_mask_newline_full, attention_mask_text], 1)
	position_ids = torch.cat([position_ids_image_full, position_ids_newline_full, position_ids_text], 1)

	# prepare the 4D attention masks for regular attention and compressv attention

	# regular attention: Q=[image_full, newline_full, text], KV=[image_full, newline_full, text]
	attention_mask_regular_4d = calculate_causal_attention_mask(position_ids, position_ids, attention_mask)

	# compressv attention: Q=[image_compress, newline_full, text], KV=[image_compress, image_full, newline_full, text]
	attention_mask_compress_4d = calculate_causal_attention_mask(torch.cat([position_ids_image_compress, position_ids_newline_full, position_ids_text], 1), torch.cat([position_ids_image_compress, position_ids_image_full, position_ids_newline_full, position_ids_text], 1), torch.cat([attention_mask_image_compress, attention_mask_image_full, attention_mask_newline_full, attention_mask_text], 1))

	min_dtype = torch.finfo(torch.bfloat16).min
	len_image_full = max_num_image_crops*per_crop_token_len
	len_image_compress = max_num_image_crops*(per_crop_token_len//compress_reduce_factor**2)
	# compress can't see full
	attention_mask_compress_4d[:, :, :len_image_compress, len_image_compress:len_image_compress+len_image_full] = min_dtype
	# others can't see compress
	attention_mask_compress_4d[:, :, len_image_compress:, :len_image_compress] = min_dtype

	return input_ids_text, labels, attention_mask, position_ids, position_ids_image_compress, attention_mask_regular_4d, attention_mask_compress_4d


def repeat_image_tokens(input_ids, labels, nums):
	image_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
	assert len(image_indices) == len(nums)

	new_input_ids = []
	new_labels = []
	image_indices = [-1] + image_indices + [input_ids.shape[0]]

	for i in range(len(image_indices) - 1):
		new_input_ids.extend(input_ids[image_indices[i] + 1 : image_indices[i + 1]].tolist())
		new_labels.extend(labels[image_indices[i] + 1 : image_indices[i + 1]].tolist())
		
		if i < len(nums):
			new_input_ids.extend([IMAGE_TOKEN_INDEX] * nums[i])
			new_labels.extend([labels[image_indices[i + 1]].item()] * nums[i])

	return torch.tensor(new_input_ids, dtype=torch.long), torch.tensor(new_labels, dtype=torch.long)


@dataclass
class DataCollatorForSupervisedDataset(object):
	"""Collate examples for supervised fine-tuning."""

	tokenizer: transformers.PreTrainedTokenizer
	max_num_image_crops: int
	per_crop_token_len: int
	compress_reduce_factor: int

	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
		max_num_image_crops = self.max_num_image_crops
		per_crop_token_len = self.per_crop_token_len
		compress_reduce_factor = self.compress_reduce_factor

		input_ids, labels = tuple([instance[key] for instance in instances]
								  for key in ("input_ids", "labels"))
		max_length = self.tokenizer.model_max_length

		padding_side = self.tokenizer.padding_side 
		assert padding_side == 'right'

		# print_rank0("Pad token id is", self.tokenizer.pad_token_id)

		image2crops_nums = [instance['image2crops_nums'] for instance in instances]

		input_ids_crops_repeated = []
		labels_crops_repeated = []
		for batch_i in range(len(input_ids)):
			cur_image2crops_nums = image2crops_nums[batch_i]
			cur_input_ids = input_ids[batch_i]
			cur_labels = labels[batch_i]
			if len(cur_image2crops_nums) > 0:
				cur_input_ids, cur_labels = repeat_image_tokens(cur_input_ids, cur_labels, cur_image2crops_nums)
			input_ids_crops_repeated.append(cur_input_ids)
			labels_crops_repeated.append(cur_labels)
				
		attention_mask = [cur_input_ids.ne(self.tokenizer.pad_token_id) for cur_input_ids in input_ids_crops_repeated]

		input_ids_text, labels, attention_mask, position_ids, position_ids_image_compress, attention_mask_regular_4d, attention_mask_compress_4d = prepare_multimodal_data(input_ids_crops_repeated, labels_crops_repeated, attention_mask, max_num_image_crops, per_crop_token_len, compress_reduce_factor, max_length, self.tokenizer.pad_token_id)
	
		batch = dict(
			input_ids=input_ids_text,
			labels=labels,
			attention_mask=attention_mask,
			attention_mask_regular_4d=attention_mask_regular_4d,
			attention_mask_compress_4d=attention_mask_compress_4d,
			position_ids=position_ids,
			position_ids_image_compress=position_ids_image_compress,
		)
		if 'images' in instances[0]:
			images_padded = []
			for instance in instances:
				images = instance['images']
				if images.shape[0] < max_num_image_crops:
					images = torch.cat([images, torch.zeros((max_num_image_crops-images, *images.shape[1:]), dtype=images.dtype)])
				images_padded.append(images)
			batch['images'] = torch.stack(images_padded)

		return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
								data_args) -> Dict:
	"""Make dataset and collator for supervised fine-tuning."""
	train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
								data_path=data_args.data_path,
								data_args=data_args)
	data_collator_kwargs = {
			'tokenizer': tokenizer,
		}

	if hasattr(data_args, 'max_num_image_crops'):
		data_collator_kwargs['max_num_image_crops'] = data_args.max_num_image_crops
	if hasattr(data_args, 'per_crop_token_len'):
		data_collator_kwargs['per_crop_token_len'] = data_args.per_crop_token_len
	if hasattr(data_args, 'compress_reduce_factor'):
		data_collator_kwargs['compress_reduce_factor'] = data_args.compress_reduce_factor


	data_collator = DataCollatorForSupervisedDataset(**data_collator_kwargs)

	return dict(train_dataset=train_dataset,
				eval_dataset=None,
				data_collator=data_collator)


# TPU Note:The TorchXLA FSDP only takes in FP32 weight. This will create an issue when you load a very large model (>30b params) on TPU in FP32. 
# TPU-V4, for example, has 100GB of memory, and a 30b model will take up at least 120GB of memory. So the solution here is to load the model in bf16.
# Then, we rewrote the FSDP sharding code to convert the bf16 weights to FP32 weights only when shard the weight. Hence, we can use minimal memory to load and shard the model on TPU.

import torch_xla
import os
XLA_DISABLE_FUNCTIONALIZATION = bool(
	os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))

def train(INDEX, attn_implementation=None):

	global local_rank
	
	log_rank0(f"Training on index {INDEX}. Local rank: {local_rank}")

	parser = transformers.HfArgumentParser(
		(ModelArguments, DataArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	local_rank = training_args.local_rank
	compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

	# verify that the train_batch_size is set correctly
	if training_args.batch_size is not None:
		if IS_XLA_AVAILABLE:
			import torch_xla.core.xla_model as xm
			world_size = xm.xrt_world_size()

			if training_args.per_device_train_batch_size is None:
				raise ValueError("If train_batch_size is set, per_device_train_batch_size must be set")

			if training_args.batch_size != training_args.per_device_train_batch_size * world_size:
				raise ValueError(f"train_batch_size ({training_args.train_batch_size}) must equal per_device_train_batch_size ({training_args.per_device_train_batch_size}) * world_size ({world_size})")

			logger.warning(f"per_device_train_batch_size is correctly set to {training_args.per_device_train_batch_size} with world_size {world_size} to match train_batch_size {training_args.batch_size}")
			logger.warning(f"train_batch_size is {training_args.train_batch_size}")

	
	# TPU Note, the original LLaMA RMSNorm implementation has a bug here, the dtype conversion is not correct. It is ok in GPU but kills TPU training.
	def forward(self, hidden_states):
		input_dtype = hidden_states.dtype
		hidden_states = hidden_states.to(torch.float32)
		variance = hidden_states.pow(2).mean(-1, keepdim=True)
		hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
		output = (self.weight * hidden_states).to(input_dtype)
		return output

	transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = forward
	transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = forward

	def new_forward_conv(self, input):
		if self.bias is None:
			return self._conv_forward(input, self.weight, self.bias)
		return self._conv_forward(input, self.weight, self.bias.to(input.dtype))

	nn.Conv2d.forward = new_forward_conv

	def new_forward_linear(self, input):
		if self.bias is None:
			return F.linear(input, self.weight, self.bias)
		return F.linear(input, self.weight, self.bias.to(input.dtype)).to(input.dtype)

	nn.Linear.forward = new_forward_linear

	bnb_model_from_pretrained_args = {}
	if training_args.bits in [4, 8]:
		from transformers import BitsAndBytesConfig
		bnb_model_from_pretrained_args.update(dict(
			device_map={"": training_args.device},
			load_in_4bit=training_args.bits == 4,
			load_in_8bit=training_args.bits == 8,
			quantization_config=BitsAndBytesConfig(
				load_in_4bit=training_args.bits == 4,
				load_in_8bit=training_args.bits == 8,
				llm_int8_skip_modules=["mm_projector"],
				llm_int8_threshold=6.0,
				llm_int8_has_fp16_weight=False,
				bnb_4bit_compute_dtype=compute_dtype,
				bnb_4bit_use_double_quant=training_args.double_quant,
				bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
			)
		))
	else:
		log_rank0(f"Loading model in full precision")

	use_cohere = False
	data_args.max_num_image_crops = model_args.max_num_image_crops
	data_args.per_crop_token_len = model_args.per_crop_token_len
	data_args.compress_reduce_factor = model_args.compress_reduce_factor

	if model_args.vision_tower_aux_list is not None:
		# Assuming model_args.model_name_or_path is a string that includes the model size
		model_name = model_args.model_name_or_path

		# Regular expression to find the number of parameters in the model's name (assuming a convention like 'ModelName-30b')
		match = re.search(r'(\d+)b', model_name)
		num_parameters_billion = float(match.group(1)) if match else 0

		# Determine if bfloat16 should be used based on the model's size
		use_bfloat16 = training_args.bf16 or num_parameters_billion > 30

		if "yi" in model_args.model_name_or_path.lower():
			use_bfloat16 = True

		elif "mistral" in model_name.lower():
			logger.warning(f"Vision tower, loading CambrianMistralForCausalLM: {model_args.model_name_or_path}")

			# replace training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer
			if (
				hasattr(training_args, 'fsdp_config') and
				'transformer_layer_cls_to_wrap' in training_args.fsdp_config.keys()
			):
				logger.warning(f"Replacing training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer. Previous value: {training_args.fsdp_config['transformer_layer_cls_to_wrap']}")
				training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["MistralDecoderLayer"]

			model = CambrianMistralForCausalLM.from_pretrained(
				model_name,
				cache_dir=training_args.cache_dir,
				do_sample=True,
				torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
				**bnb_model_from_pretrained_args
			)
			transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = forward
		elif "qwen" in model_name.lower():
			logger.warning(f"Vision tower, loading CambrianQwenForCausalLM: {model_args.model_name_or_path}")

			# replace training_args.fsdp_config.transformer_layer_cls_to_wrap with Qwen2lDecoderLayer
			if (
				hasattr(training_args, 'fsdp_config') and
				'transformer_layer_cls_to_wrap' in training_args.fsdp_config.keys()
			):
				logger.warning(f"Replacing training_args.fsdp_config.transformer_layer_cls_to_wrap with Qwen2DecoderLayer. Previous value: {training_args.fsdp_config['transformer_layer_cls_to_wrap']}")
				training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["Qwen2DecoderLayer"]

			model = CambrianQwenForCausalLM.from_pretrained(
				model_name,
				cache_dir=training_args.cache_dir,
				do_sample=True,
				torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
				**bnb_model_from_pretrained_args
			)
			transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = forward
		elif "phi-3" in model_name.lower():
			logger.warning(f"Vision tower, loading CambrianPhi3ForCausalLM: {model_args.model_name_or_path}")
			
			# replace training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer
			if (
				hasattr(training_args, 'fsdp_config') and
				'transformer_layer_cls_to_wrap' in training_args.fsdp_config.keys()
			):
				logger.warning(f"Replacing training_args.fsdp_config.transformer_layer_cls_to_wrap with Phi3DecoderLayer. Previous value: {training_args.fsdp_config['transformer_layer_cls_to_wrap']}")
				training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["Phi3DecoderLayer"]
			model = CambrianPhi3ForCausalLM.from_pretrained(
					model_name,
					cache_dir=training_args.cache_dir,
					do_sample=True,
					torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
					**bnb_model_from_pretrained_args
			)
			cambrian.model.language_model.phi3.modeling_phi3.Phi3RMSNorm.forward = forward
		else:
			logger.warning(f"Vision tower, loading CambrianLlamaForCausalLM: {model_args.model_name_or_path}")
			model = CambrianLlamaForCausalLM.from_pretrained(
				model_args.model_name_or_path,
				cache_dir=training_args.cache_dir,
				do_sample=True,
				torch_dtype=(torch.bfloat16 if use_bfloat16 else None),
				**bnb_model_from_pretrained_args
			)
	else:
		logger.warning(f"No vision tower, loading pure language model: {model_args.model_name_or_path}")
		model = transformers.LlamaForCausalLM.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=training_args.cache_dir,
			attn_implementation=attn_implementation,
			torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
			**bnb_model_from_pretrained_args
		)
	model.config.use_cache = False
	model.generation_config.do_sample = True

	if model_args.freeze_backbone:
		model.model.requires_grad_(False)

	log_rank0("Model loaded.")

	if training_args.bits in [4, 8]:
		from peft import prepare_model_for_kbit_training
		model.config.torch_dtype = (
			torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
		model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

	if training_args.gradient_checkpointing:
		log_rank0("Using gradient checkpointing")
		if hasattr(model, "enable_input_require_grads"):
			model.enable_input_require_grads()
		else:
			def make_inputs_require_grad(module, input, output):
				output.requires_grad_(True)

			model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

	if training_args.lora_enable:
		log_rank0("Adding LoRA adapters...")
		from peft import LoraConfig, get_peft_model
		lora_config = LoraConfig(
			r=training_args.lora_r,
			lora_alpha=training_args.lora_alpha,
			target_modules=find_all_linear_names(model),
			lora_dropout=training_args.lora_dropout,
			bias=training_args.lora_bias,
			task_type="CAUSAL_LM",
		)
		if training_args.bits == 16:
			if training_args.bf16:
				model.to(torch.bfloat16)
			if training_args.fp16:
				model.to(torch.float16)
		print_rank0("Adding LoRA adapters...")
		model = get_peft_model(model, lora_config)

	log_rank0("Configuring tokenizer...")
	if 'mpt' in model_args.model_name_or_path or 'qwen' in model_args.model_name_or_path.lower():
		tokenizer = transformers.AutoTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=training_args.cache_dir,
			model_max_length=training_args.model_max_length,
			padding_side="right"
		)
	else:
		tokenizer = transformers.AutoTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=training_args.cache_dir,
			model_max_length=training_args.model_max_length,
			padding_side="right",
			use_fast=False,
		)

	if model_args.version == "v0":
		if tokenizer.pad_token is None:
			smart_tokenizer_and_embedding_resize(
				special_tokens_dict=dict(pad_token="[PAD]"),
				tokenizer=tokenizer,
				model=model,
			)
	elif model_args.version == "v0.5":
		tokenizer.pad_token = tokenizer.unk_token
	elif model_args.version == "llama_v3":
		tokenizer.pad_token = "<|reserved_special_token_0|>"
		tokenizer.pad_token_id = 128002
	elif "qwen" in model_args.version:
		tokenizer.pad_token = "<|endoftext|>"
		tokenizer.pad_token_id = 151643
	else:
		tokenizer.pad_token = tokenizer.unk_token
		if model_args.version in conversation_lib.conv_templates:
			conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
		else:
			logger.warning(f"Conversation version {model_args.version} not found. Using default `vicuna_v1`")
			conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

	# log_rank0(f"Default conversation version: {conversation_lib.default_conversation.version}")
	# print_rank0("Then it is", conversation_lib.default_conversation)

	if use_cohere:
		tokenizer.pad_token_id = 0
		print_rank0("tokenizer id is", tokenizer.pad_token_id)
	# print_rank0("tokenizer is", tokenizer)

	if model_args.vision_tower_aux_list is not None:
		model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
		model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
		model_args.vision_tower_aux_token_len_list = json.loads(model_args.vision_tower_aux_token_len_list)
		model_args.query_num_list = json.loads(model_args.query_num_list)
		model.get_model().initialize_vision_modules(
			model_args=model_args,
			fsdp=training_args.fsdp
		)
		model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower

		vision_tower_aux_list = None
		if model_args.vision_tower_aux_list is not None:
			vision_tower_aux_list = model.get_vision_tower_aux_list()
		
		if not training_args.unfreeze_mm_vision_tower:
			# vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
			if vision_tower_aux_list is not None:
				for vision_tower_aux in vision_tower_aux_list:
					vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
		else:
			# vision_tower.to(device=training_args.device)
			if vision_tower_aux_list is not None:
				for vision_tower_aux in vision_tower_aux_list:
					vision_tower_aux.to(device=training_args.device)
				# vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
		# data_args.image_processor = vision_tower.image_processor
		if vision_tower_aux_list is not None:
			data_args.image_processor_aux_list = [vision_tower_aux.image_processor for vision_tower_aux in vision_tower_aux_list]
		data_args.is_multimodal = True

		model.config.image_aspect_ratio = data_args.image_aspect_ratio
		model.config.tokenizer_padding_side = tokenizer.padding_side
		model.config.tokenizer_model_max_length = tokenizer.model_max_length

		model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
		if model_args.tune_mm_mlp_adapter:
			model.requires_grad_(False)
			# for p in model.get_model().mm_projector.parameters():
			#     p.requires_grad = True
			tune_modules = ['mm_projector', 'pos_emb', 'vision_sampler', 'vision_sampler_layers', 'vision_query', 'image_newline', 'vision_mlp_layers']
			for name, param in model.named_parameters():
				if any(listed_name in name for listed_name in tune_modules):
					print_rank0('tuning {}'.format(name))
					param.requires_grad = True

		model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
		if training_args.freeze_mm_mlp_adapter:
			for p in model.get_model().mm_projector.parameters():
				p.requires_grad = False
		if training_args.unfreeze_mm_vision_tower:
			if vision_tower_aux_list is not None:
				for vision_tower_aux in vision_tower_aux_list:
					for p in vision_tower_aux.parameters():
						p.requires_grad = True

		if training_args.bits in [4, 8]:
			log_rank0(f"Initializing vision modules in {training_args.bits}bit")
			model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

		model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
		model.config.mm_projector_lr = training_args.mm_projector_lr
		model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr
		model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
		model.config.mm_vision_mlp_lr = training_args.mm_vision_mlp_lr
		training_args.use_im_start_end = model_args.mm_use_im_start_end
		model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
		model.config.vision_tower_aux_token_len_list = data_args.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
		model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

	if training_args.bits in [4, 8]:
		log_rank0(f"Initializing model in {training_args.bits}bit")
		from peft.tuners.lora import LoraLayer
		for name, module in model.named_modules():
			if isinstance(module, LoraLayer):
				if training_args.bf16:
					module = module.to(torch.bfloat16)
			if 'norm' in name:
				module = module.to(torch.float32)
			if 'lm_head' in name or 'embed_tokens' in name:
				if hasattr(module, 'weight'):
					if training_args.bf16 and module.weight.dtype == torch.float32:
						module = module.to(torch.bfloat16)

	log_rank0("Configuring data module...")
	data_module = make_supervised_data_module(tokenizer=tokenizer,
											  data_args=data_args)

	

	if training_args.bf16:
		model = model.to(dtype=torch.float32)

	callbacks = []

	# if "wandb" in training_args.report_to:
	#     wandb_nan_callback = NanInfAlertWandbCallback(metrics=["loss"])
	#     callbacks.append(wandb_nan_callback)
	#     # rm wandb from training_args.report_to so it doesn't get passed to the Trainer
	#     training_args.report_to.remove("wandb")
	#     assert "wandb" not in training_args.report_to, training_args.report_to


	log_rank0("Configuring trainer...")
	trainer = CambrianTrainer(model=model,
					tokenizer=tokenizer,
					args=training_args,
					# extra_losses=["lm_loss", "aux_loss"],
					**data_module)
	trainer.is_fsdp_enabled = True
	if training_args.train_continue:
		resume_from_checkpoint=training_args.resume_from_checkpoint
		trainer.train(resume_from_checkpoint=resume_from_checkpoint)
	else:
		trainer.train()

	log_rank0(f"Training finished: {training_args.output_dir}")
	
	trainer.save_state()

	model.config.use_cache = True

	log_rank0("Saving model...")
	if training_args.lora_enable:
		state_dict = get_peft_state_maybe_zero_3(
			model.named_parameters(), training_args.lora_bias
		)
		non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
			model.named_parameters()
		)
		if training_args.local_rank == 0 or training_args.local_rank == -1:
			model.config.save_pretrained(training_args.output_dir)
			model.save_pretrained(training_args.output_dir, state_dict=state_dict)
			torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
	else:
		safe_save_model_for_hf_trainer(trainer=trainer,
									   output_dir=training_args.output_dir)


if __name__ == "__main__":
	train()
