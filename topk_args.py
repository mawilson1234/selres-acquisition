# check_args.py
#
# check stats for candidate arguments for use in new verb experiments using a dataset which
# have strings that are tokenized as single words in all of the model types in conf/model
import hydra
import torch

import pandas as pd

import torch.nn.functional as F

from typing import *
from omegaconf import DictConfig, OmegaConf

from scipy.stats import pearsonr
from transformers import logging as lg
from transformers import AutoTokenizer, AutoModelForMaskedLM
lg.set_verbosity_error()

OmegaConf.register_new_resolver(
	'model_name', 
	lambda string_id: string_id.replace('google/', '').replace('-seed', '')
)

OmegaConf.register_new_resolver(
	'which_args',
	lambda string_id, which_args: which_args if not which_args == 'model' else string_id.replace('google/', '').replace('-seed', '')
)

@hydra.main(config_path='conf', config_name='selres')
def topk_args(cfg: DictConfig) -> None:
	'''
	Generates candidate nouns that a model considers most likely in different argument positions. Saves results to disk
	
		params:
			cfg (dictconfig)	: a configuration specifying sentence data to use, as well as target frequency for candidate nouns
	'''
	print(OmegaConf.to_yaml(cfg, resolve=True))
	
	if 'multibert' in cfg.model.string_id:
		tokenizer = AutoTokenizer.from_pretrained(cfg.model.string_id, **cfg.model.tokenizer_kwargs)
	else:
		tokenizer = AutoTokenizer.from_pretrained(cfg.model.string_id.replace('-base', '-base-1B'), **cfg.model.tokenizer_kwargs)
	mask_token = tokenizer.mask_token
	
	data, masked_data = load_data(cfg)
	
	predictions = get_topk_predictions(cfg, data, masked_data, tokenizer)
	save_predictions(predictions)
	
def get_topk_predictions(
	cfg: DictConfig, 
	data: List[str],
	masked_data: List[str],
	tokenizer: 'PreTrainedTokenizer',
) -> pd.DataFrame:
	if not cfg.data.which_args == 'model':
		args = cfg.data.which_args 
	else:
		if 'multibert' in cfg.model.string_id:
			args = cfg.model.string_id.replace('google/', '').replace('-seed', '')
		else:
			args = cfg.model.string_id.replace('nyu-mll/', '').replace('-base', '-base-1B')
	
	gfs = set([gf for verb in cfg.data[args] for voice in cfg.data[args][verb] for gf in cfg.data[args][verb][voice]])
	inputs = tokenizer(masked_data, return_tensors='pt', padding=True)
	
	# We need to get the order/positions of the arguments for each sentence 
	# in the masked data so that we know which argument we are pulling 
	# out predictions for, because when we replace the argument placeholders 
	# with mask tokens, we lose information about which argument corresponds to which mask token
	gfs_in_order 			= [[word for word in sentence.replace('.', '').split() if word in gfs] for sentence in data]
	masked_indices 			= [[index for index, token_id in enumerate(i) if token_id == tokenizer.convert_tokens_to_ids(tokenizer.mask_token)] for i in inputs['input_ids']]
	sentence_gf_indices 	= [dict(zip(gf, index)) for gf, index in tuple(zip(gfs_in_order, masked_indices))]
	
	# Run the model on the masked inputs to get the predictions
	model = AutoModelForMaskedLM.from_pretrained(cfg.model.string_id, **cfg.model.model_kwargs)
	model.eval()
	with torch.no_grad():
		outputs = model(**inputs)
	
	# Convert predicted logits to probabilities
	probs = F.softmax(outputs.logits, dim=-1)
	
	predictions = []
	# get the topk non overlapping predictions for each arg position in each sentence
	for sentence, sentence_type, gf_indices, prob in zip(data, cfg.data.sentence_types, sentence_gf_indices, probs):
		for gf, index in gf_indices.items():
			other_gf_indices = {k: v for k, v in gf_indices.items() if not k == gf}
			to_filter = set([x for _, other_index in other_gf_indices.items() for x in tokenizer.convert_ids_to_tokens(torch.topk(prob[other_index], k=cfg.num_args).indices)])
			topk = tokenizer.convert_ids_to_tokens(torch.topk(prob[index], k=cfg.num_args, sorted=True).indices)
			more = 0
			while any(k in to_filter for k in topk) or not len(topk) == cfg.num_args:
				more += 1
				to_filter = set([x for _, other_index in other_gf_indices.items() for x in tokenizer.convert_ids_to_tokens(torch.topk(prob[other_index], k=cfg.num_args+more).indices)])
				topk = [k for k in tokenizer.convert_ids_to_tokens(torch.topk(prob[index], k=cfg.num_args+more, sorted=True).indices) if not k in to_filter][:cfg.num_args]
			
			predictions.append({
				'model': cfg.model.string_id.replace('google/', '').replace('-seed', '').replace('nyu-mll/', '').replace('-base-', '_'),
				'sentence': sentence,
				'verb': sentence_type.split()[0],
				'voice': sentence_type.split()[1],
				'gf': gf,
				'topk': ', '.join(topk)
			})
	
	return pd.DataFrame(predictions)

def load_data(cfg: DictConfig) -> Tuple[List[str]]:
	lines = [line.strip() for line in cfg.data.data]
	lines = [l.lower() for l in lines]
	
	if not cfg.data.which_args == 'model':
		args = cfg.data.which_args 
	else:
		if 'multibert' in cfg.model.string_id:
			args = cfg.model.string_id.replace('google/', '').replace('-seed', '')
		else:
			args = cfg.model.string_id.replace('nyu-mll/', '').replace('-base', '-base-1B')
	
	gfs = set([gf for verb in cfg.data[args] for voice in cfg.data[args][verb] for gf in cfg.data[args][verb][voice]])
	
	mask_token = AutoTokenizer.from_pretrained(cfg.model.string_id, **cfg.model.tokenizer_kwargs).mask_token
	
	masked_data = []
	for l in lines:
		for gf in gfs:
			l = l.replace(gf, mask_token)
		
		masked_data.append(l)
	
	return lines, masked_data

def save_predictions(predictions: List[Dict]):
	file = f'{predictions.model.unique()[0]}:\n'
	for verb, df in predictions.groupby('verb', sort=False):
		file += f'  {verb}:\n'
		for voice, df2 in df.groupby('voice', sort=False):
			file += f'    {voice}:\n'
			for gf, df3 in df2.groupby('gf', sort=False):
				file += f'      "{gf}": [{df3.topk.unique()[0]}]\n'
	
	with open(f'{predictions.model.unique()[0]}-args.yaml', 'wt') as out_file:
		out_file.write(file)

if __name__ == '__main__':
	
	topk_args()
