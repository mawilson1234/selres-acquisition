# selres.py
# 
# Tunes a model on training data and provides functions for evaluation
import os
import re
import hydra
import torch
from torch.distributions import Categorical
import logging

import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from copy import deepcopy
from typing import *
from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig

from transformers import logging as lg
lg.set_verbosity_error()
from transformers import AutoModelForMaskedLM, AutoTokenizer

from . import selres_plots
from . import selres_utils

log = logging.getLogger(__name__)

ALL_CHECKPOINTS = {
	'multiberts': list(range(0,200,20)) + list(range(200,2001,100)),
	'miniberta': ['med-small-1M', 'base-10M', 'base-100M', 'base-1B']
}

LOWER = {
	'multiberts': True,
	'miniberta': False,
}

class SelectionalRestrictionEvaluator:
	
	# START Private Functions
	
	# during tuning
	def __format_dataset(self, dataset: Union[Dict, DictConfig] = None) -> Dict:
		'''
		Returns a dictionary with formatted inputs, labels, and masked_indices for a dataset
		
			params:
				dataset (Dict-like)		: the dataset to generated formatted data for
			
			returns:
				a dict with sentences, inputs, masked_indices
		'''
		gf_regex = re.sub(r'(\[|\])', '\\ \\1', '|'.join(set([gf for verb in self.cfg.data.args for voice in self.cfg.data.args[verb] for gf in self.cfg.data.args[verb][voice]]))).replace(' ', '')
		masked_args	= [re.findall(rf'({gf_regex})', sentence) for sentence in dataset]
		for i, _ in enumerate(dataset):
			for gf in set([gf for verb in self.cfg.data.args for voice in self.cfg.data.args[verb] for gf in self.cfg.data.args[verb][voice]]):
				dataset[i] = dataset[i].replace(gf, self.mask_token)
		
		# this is so we don't overwrite the original dataset as we do this
		dataset = deepcopy(dataset)
		inputs = selres_utils.format_data_for_tokenizer(dataset, mask_token=self.mask_token, lower=lower[self.model_base])
		inputs = self.tokenizer(inputs, return_tensors='pt', padding=True).to(self.device)
		
		masked_indices = [(sentence == self.mask_token_id).nonzero(as_tuple=True)[0].tolist() for sentence in inputs['input_ids']]
		gf_masked_indices = [dict(zip(args, indices)) for indices, args in zip(masked_indices, masked_args)]
		
		formatted_data = {'sentences': dataset, 'inputs': inputs, 'masked_indices': gf_masked_indices}
		
		return formatted_data
	
	# formatting of results
	def __add_hyperparameters_to_summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
		'''
		Adds hyperparameters to a summary dataframe.
			
			params:
				df (pd.DataFrame): a dataframe to add hyperparameters to
			
			returns:
				df (pd.DataFrame): the dataframe with hyperparameters added in columns
		'''
		exclude = [
			'mask_token', 'mask_token_id', 'unk_token', 'unk_token_id', 'device',
		]
		
		included_vars = [var for var in vars(self) if not var in exclude]
		included_vars = [var for var in included_vars if isinstance(vars(self)[var],(str,int,float,bool))]
		sorted_vars = sorted([var for var in included_vars], key=lambda item: re.sub(r'^(model)', '0\\1', item))
		
		for var in sorted_vars:
			df[var] = vars(self)[var]
		
		df['string_id'] = self.cfg.model.string_id	
		df['data'] = self.cfg.data.name
		df['args_group'] = self.cfg.data.which_args \
							if not self.cfg.data.which_args == 'model' \
							else self.cfg.model.string_id.replace('google/', '').replace('-seed', '').replace('nyu-mll/', '').replace('-base', '_')
		
		return df
	
	# evaluation
	def __collect_results(
		self, 
		outputs: 'MaskedLMOutput',
		sentences: List[str],
		sentence_types: List[str],
		sentence_nums: List[int],
		masked_indices: List[Dict[str,int]],
	) -> List:
		'''
		Returns a list of dicts with results based on model outputs
			
			params:
				outputs (MaskedLMOutput): model outputs
				sentences (list)		: the sentences that were used to generate the outputs.
				sentence_types (list)	: a list of sentence_types. used to determine which arguments to evaluate for each sentence
				sentence_nums (list)	: a list of sentence numbers used within each sentence type
				masked_indices (list)	: list of dicts with mappings from string token to integer positions 
										  of the masked token locations corresponding to that token 
										  for each sentence in the outputs
			
			returns:
				list of dicts with results for each token for each sentence
		'''
		def get_output_metrics(outputs: 'MaskedLMOutput') -> Tuple:
			logits 				= outputs.logits
			probabilities 		= F.softmax(logits, dim=-1)
			log_probabilities 	= F.log_softmax(logits, dim=-1)
			surprisals 			= -(1/torch.log(torch.tensor(2.))) * F.log_softmax(logits, dim=-1)
			predicted_ids 		= torch.argmax(log_probabilities, dim=-1)
			
			return logits, probabilities, log_probabilities, surprisals, predicted_ids
		
		results = []
		metrics = zip(masked_indices, sentences, sentence_types, sentence_nums, *get_output_metrics(outputs))
		
		for position_indices, sentence, sentence_type, sentence_num, logits, probs, logprobs, surprisals, predicted_ids in metrics:
			verb = sentence_type.split()[0]
			voice = sentence_type.split()[1]
			verb_profile = self.cfg.data.verb_profiles[verb]
			
			tokens_to_type_labels = {token: label for label in self.cfg.data.args[verb][voice] for token in self.cfg.data.args[verb][voice][label]}
			if len(tokens_to_type_labels.keys()) < len(selres_utils.flatten(list(self.cfg.data.args[verb][voice].values()))):
				raise ValueError('Tokens should only be used in a single eval group!')
			
			lin_positions 	= sorted(list(position_indices.keys()), key=lambda position: position_indices[position])
			lin_positions 	= {p: lin_positions.index(p) + 1 for p in lin_positions}
							
			for position in position_indices:
				for token, label in tokens_to_type_labels.items():
					token_id = self.tokenizer.convert_tokens_to_ids(token)
					
					if token_id == self.unk_token_id:
						raise ValueError(f'Token "{token}" was not tokenized correctly! Try using something different instead.')
					
					entropy 	= Categorical(probs=probs[position_indices[position]]).entropy()
					
					# get the lobprob for the position
					logprob 	= logprobs[position_indices[position],token_id]
					
					common_args = {
						'arg type'			: label,
						'token id'			: token_id,
						'token'				: token,
						'position'			: position,
						'sentence'			: sentence,
						'sentence type'		: sentence_type,
						'verb'				: verb,
						'verb profile'		: verb_profile,
						'voice'				: voice,
						'sentence num'		: sentence_num,
						'predicted sentence': self.tokenizer.decode(predicted_ids),
						'predicted ids'		: ' '.join([str(i.item()) for i in predicted_ids]),
						'logit'				: logits[position_indices[position],token_id],
						'probability'		: probs[position_indices[position],token_id],
						'log probability'	: logprob,
						'surprisal'			: surprisals[position_indices[position],token_id],
						'args group'		: self.cfg.data.which_args if not self.cfg.data.which_args == 'model' else self.cfg.model.string_id.replace('google/', '').replace('-seed', ''),
						'entropy'			: entropy
					}
					
					# we only want to consider other masked positions that contain tokens in the eval groups for odds ratios
					other_positions = [
						(other_label, index)
						for other_label, index in position_indices.items()
							if not other_label == position
					]
					
					if other_positions:
						for other_position, index in other_positions:
							# we compare a token to itself in the other position
							# this addresses overall probability biases due to e.g. frequency and length
							logprob2 		= logprobs[index,token_id]
							
							results.append({
								'odds ratio'		: logprob - logprob2,
								'gf ratio name'		: f'{position}/{other_position}',
								'linear ratio name'	: f'position {lin_positions[position]}/position {lin_positions[other_position]}',
								**common_args
							})
					else:
						results.append({**common_args})
		
		return results
	
	# END Private Functions
	
	# START Class Functions
	
	def __init__(self, cfg: DictConfig):
		'''
		Creates a selres object, loads argument/dev sets, and sets class attributes
		
			params:
				cfg (DictConfig): a dictconfig specifying a selres configuration
				use_gpu	(bool)	: used during evaluation. useful when loading a model trained on cpu on gpu, or vice versa
		'''
		def check_tokenizations(tokens: Set[str]):
			tokenized = [self.tokenizer.tokenize(token) for token in tokens]
			if any(tokens[0] == self.unk_token or len(tokens) > 1 for tokens in tokenized):
				raise ValueError(f'Some arguments were not tokenized as a single token!: {[token for token, tokenized_token in zip(tokens, tokenized) if tokenized_token[0] == self.unk_token or len(tokenized_token) > 1]}')
		
		self.cfg = cfg
		
		# too little memory to use gpus locally, but we can specify to use them on the cluster with use_gpu=true
		self.device	= 'cuda' if torch.cuda.is_available() and self.cfg.use_gpu else 'cpu'
		
		if self.device == 'cuda':
			log.info(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
		
		with open_dict(self.cfg):
			self.cfg.data.args = self.cfg.data[self.cfg.data.which_args] if not self.cfg.data.which_args == 'model' else self.cfg.data[self.cfg.model.string_id.replace('google/', '').replace('-seed', '')]
		
		self.model_base = 'multiberts' if 'multibert' in self.cfg.model.string_id else 'miniberta'
		
		self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.string_id, **self.cfg.model.tokenizer_kwargs)
		self.mask_token	= self.tokenizer.mask_token
		self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
		self.unk_token = self.tokenizer.unk_token
		self.unk_token_id = self.tokenizer.convert_tokens_to_ids(self.unk_token)
		
		self.data = self.load_data()
		
		breakpoint()

		all_args = set([arg for verb in self.cfg.data.args for voice in self.cfg.data.args[verb] for gf in self.cfg.data.args[verb][voice] for arg in self.cfg.data.args[verb][voice][gf]])
		check_tokenizations(all_args)		

		self.all_checkpoints = [f'{self.cfg.model.string_id}-step_{checkpoint}k' for checkpoint in ALL_CHECKPOINTS[self.model_base]] \
			if self.model_base == 'multiberts' \
			else [self.cfg.model.string_id.replace('base', checkpoint) for checkpoint in ALL_CHECKPOINTS[self.model_base]]
	
	def __repr__(self) -> str:
		'''Return a string that eval() can be called on to create an identical selres object'''
		return 'selres.SelectionalRestrictionsEvaluator(' + repr(self.cfg) + ')'
	
	def __str__(self) -> str:
		'''Return a formatted string for printing'''
		return f'SelectionalRestrictionsEvaluator object @ {self.checkpoint_dir} with config:\n' + OmegaConf.to_yaml(self.cfg, resolve=True)
	
	# END Class Functions
	
	# evaluation functions
	def load_data(self) -> Dict:
		'''
		Loads the data in self.cfg.data.
		
			returns:
				dict with sentences, inputs, and masked indices for each sentence type in the data file
		'''
		lines					= [line.strip() for line in self.cfg.data.data]
		
		sentences 				= [[s.strip() for s in r.split(' , ')] for r in lines]
		sentences 				= selres_utils.format_data_for_tokenizer(sentences, mask_token=self.mask_token, lower=lower[self.model_base])
		sentence_types 			= self.cfg.data.sentence_types
		
		lens 					= [len(sentence_group) for sentence_group in sentences]
		
		# faster to flatten the inputs and then restore instead of looping
		flattened_sentences 	= selres_utils.flatten(sentences)
		inputs_dict				= self.__format_dataset(dataset=flattened_sentences)
		
		# unpack the results to group by sentence type
		masked_inputs 			= inputs_dict['inputs']
		masked_indices 			= inputs_dict['masked_indices']
		
		types_sentences 		= {}
		for sentence_type, n_sentences, sentence_group in zip(sentence_types, lens, sentences):
			types_sentences[sentence_type]								= {}
			types_sentences[sentence_type]['sentences'] 				= sentence_group
			
			types_sentences[sentence_type]['inputs']					= {}
			types_sentences[sentence_type]['inputs']['input_ids']		= masked_inputs['input_ids'][:n_sentences]
			types_sentences[sentence_type]['inputs']['attention_mask']	= masked_inputs['attention_mask'][:n_sentences]
			masked_inputs['input_ids']									= masked_inputs['input_ids'][n_sentences:]
			masked_inputs['attention_mask']								= masked_inputs['attention_mask'][n_sentences:]
			
			types_sentences[sentence_type]['masked_indices']			= masked_indices[:n_sentences]
			masked_indices												= masked_indices[n_sentences:]
		
		if (
			not all(sentence_type in types_sentences for sentence_type in sentence_types) or \
			any(masked_indices) or torch.any(masked_inputs['input_ids']) or torch.any(masked_inputs['attention_mask'])
		):
			raise ValueError('The number of sentences and inputs does not match!')
		
		return types_sentences
	
	def evaluate(self) -> None:
		'''Computes model performance'''
		summary = self.get_summary()
		
		# convert tensors to float before saving
		for c in summary.columns:
			if isinstance(summary[c][0],torch.Tensor):
				summary[c] = summary[c].astype(float)
		
		file_prefix = selres_utils.get_file_prefix(summary)
		
		log.info(f'Saving to "{os.getcwd()}"')
		summary.to_csv(f'{file_prefix}-scores.csv.gz', index=False, na_rep='NaN')
		
		if self.cfg.create_plots:
			log.info('Plotting learning curves')
			self.plot_learning_curves(summary)
		
		log.info('Evaluation complete')
		print('')
	
	def get_summary(self) -> pd.DataFrame:
		'''Returns a dataframe containing a summary of predictions for the dataset'''
		
		# when we load the data, we want to return it grouped by sentence type for general ease of use.
		# however, concatenating everything together for evaluation is faster. For this reason, we join everything together,
		# then evaluate, and then split it apart. this may seem a bit redundant, but that's why we're doing it this way
		inputs = {
			'input_ids'		: torch.cat([self.data[sentence_type]['inputs']['input_ids'] for sentence_type in self.data]),
			'attention_mask': torch.cat([self.data[sentence_type]['inputs']['attention_mask'] for sentence_type in self.data])
		}
		
		masked_indices			= selres_utils.flatten([self.data[sentence_type]['masked_indices'] for sentence_type in self.data])
		sentences				= selres_utils.flatten([self.data[sentence_type]['sentences'] for sentence_type in self.data])
		
		# get these to add back to the flattened data
		num_sentences			= [len(self.data[sentence_type]['sentences']) for sentence_type in self.data]
		sentence_nums			= selres_utils.flatten([list(range(num)) for num in num_sentences])
		sentence_types			= [sentence_type for sentence_type in self.data]
		sentence_types			= selres_utils.flatten([[sentence_type] * num for sentence_type, num in zip(sentence_types, num_sentences)])
		
		summary 				= []
		with logging_redirect_tqdm():
			for string_id in tqdm(self.all_checkpoints):
				log.info(f'Evaluating model checkpoint: {string_id}')
				model 		= AutoModelForMaskedLM.from_pretrained(string_id, **self.cfg.model.model_kwargs).to(self.device)
				model.eval()
				
				with torch.no_grad():
					outputs 		= model(**inputs)
				
				checkpoint_results	= self.__collect_results(outputs=outputs, sentences=sentences, sentence_nums=sentence_nums, sentence_types=sentence_types, masked_indices=masked_indices)
				formatted_checkpoint = selres_utils.format_checkpoint(string_id)
				checkpoint_results = [{**d, 'checkpoint': formatted_checkpoint} for d in checkpoint_results]
				summary.extend(checkpoint_results)
		
		summary 		= pd.DataFrame(summary)
		summary['token accuracy'] = summary['odds ratio'] > 0
		
		# get confidence scores for each group for each checkpoint
		if 'gf ratio name' in summary.columns:
			summary['gf ratio conf'] = np.nan
			for (checkpoint, sentence), df in summary.groupby(['checkpoint', 'sentence']):
				for arg_type, df2 in df.groupby('arg type'):
					df2 = df2[df2['gf ratio name'].str.startswith(arg_type)]
					for ratio_name, df3 in df2.groupby('gf ratio name'):
						mean_prob = df3.probability.sum()
						other_arg_types = df[df['arg type'] != arg_type]['arg type'].unique()
						for other_arg_type in other_arg_types:
							other_mean_prob = df[(df['arg type'] == other_arg_type) & (df['gf ratio name'] == f'{arg_type}/{other_arg_type}')].probability.sum()
							group_conf = np.log(mean_prob/other_mean_prob)
							summary.loc[
									(summary.checkpoint == checkpoint) & 
									(summary.sentence == sentence) & 
									(summary['arg type'] == arg_type) &
									(summary['gf ratio name'] == ratio_name),
									'gf ratio conf'
								] = float(group_conf)
			
			summary['grammatical function accuracy'] = summary['gf ratio conf'] > 0
				
		# we only needed the rows with probabilities for e.g. objects in subject position temporarily to get the confidence score
		# now we can drop the rows for which there is no confidence score to get just what we want
		summary 		= summary.dropna().reset_index(drop=True)
		
		# reorder and rename the columns for display and ease of use
		summary			= selres_utils.move_cols(summary, cols_to_move=['sentence type', 'sentence num'], ref_col='sentence', position='after')
		summary.columns = [col.replace(' ', '_') for col in summary.columns]
		
		summary = self.__add_hyperparameters_to_summary_df(summary)
		
		# for now, we are not using these columns, so we're dropping them before returning. we can easily change this later if desired
		summary = summary.drop(
			['logit', 'probability', 'log_probability', 'surprisal', 'predicted_sentence', 'predicted_ids'], axis=1
		)
		
		return summary
	
	# wrapper/helper functions for plots/accuracies (implemented in selres_plots)
	def plot_learning_curves(self, summary: pd.DataFrame) -> None:
		
		'''
		Calls selres_plots.create_odds_ratios_plots
		
			params:
				summary (pd.DataFrame): passed to selres_plots.plot_learning_curves
		'''
		selres_plots.plot_learning_curves(summary)
