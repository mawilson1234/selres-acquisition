# tuner_utils.py
#
# utility functions for tuner.py
import re
import torch

import numpy as np
import pandas as pd

from copy import deepcopy
from typing import *
from omegaconf import OmegaConf, DictConfig, ListConfig

# short useful functions
def flatten(l: List) -> List:
	'''
	Flatten a list of lists or np.ndarrays without breaking strings into characters
	adapted from https://stackoverflow.com/questions/5286541/how-can-i-flatten-lists-without-splitting-strings
	
		params:
			l (list): a list to flatten
		
		returns:
			l (list): the list, flattened into 1 dimension
	'''
	l = apply_to_all_of_type(l, ListConfig, OmegaConf.to_container)
	if l is not None:
		return [k for j in ([i] if not isinstance(i,(list,tuple,np.ndarray)) else flatten(i) for i in l) for k in j]

def listify(l: 'any') -> List:
	'''
	If something isn't a list, return it as a list. If it is, return it without changing it.
		
		params:
			l (any)	: an object that we want to ensure is a list
		
		returns:
			l (list): the object as/in a list
	'''
	if isinstance(l,list):
		return l
	
	if isinstance(l,ListConfig):
		return OmegaConf.to_container(l)
	
	if isinstance(l,tuple):
		return list(l)
	
	if isinstance(l,(np.ndarray,pd.Series)):
		return l.tolist()
	
	return [l]

def apply_to_all_of_type(
	data: 'any', 
	t: Type,
	fun: Callable,
	*args: Tuple, 
	**kwargs: Dict
) -> 'any':
	'''
	Apply a function to recursively to all elements in an iterable that match the specified type
		
		params:
			data (any)		: an object to recursively apply a function to
			t (type)		: the type of object within data to apply the function to
			fun (Callable)	: the function to apply to any values within data of type t
			*args (tuple)	: passed to fun
			**kwargs (dict)	: passed to fun
		
		returns:
			data (any)		: the data object with fun applied to everything in data matching type t
	'''
	if isinstance(data,(DictConfig,ListConfig)):
		# we need the primitive versions of these so we can modify them
		data = OmegaConf.to_container(data)
	
	data = deepcopy(data)
	
	if isinstance(data,t):
		returns = fun(data, *args, **kwargs)
	elif isinstance(data,dict):
		returns = {apply_to_all_of_type(k, t, fun, *args, **kwargs): apply_to_all_of_type(v, t, fun, *args, **kwargs) for k, v in data.items()}
	elif isinstance(data,(list,tuple,set)):
		returns = type(data)(apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data)
	elif isinstance(data,(torch.Tensor,pd.Series)):
		returns = type(data)([apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data])
	elif isinstance(data,np.ndarray):
		returns = np.array([apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data])
	else:
		returns = data
	
	if isinstance(data,(pd.Series,np.ndarray)):
		return returns if returns.any() or returns.size == 1 and returns[0] == 0 else None
	else:
		return returns

def crawler(t: 'type') -> Callable:
	'''
	Creates functions that crawl through a nested data structure 
	and apply a transformation to a single data type.
		
		params:
			t (type)		: the type to apply the function to
	'''
	return lambda fun: \
		lambda data, *args, **kwargs: \
			apply_to_all_of_type(data=data, t=t, fun=fun, *args, **kwargs)

# summary and file related
def get_file_prefix(summary: pd.DataFrame) -> str:
	'''
	Creates an appropriate file prefix for saving various output files
	
		params:
			summary (pd.DataFrame)	: a summary dataframe containing information about the experiment's configuration
		
		returns:
			file_prefix (str)		: a string containing the dataset and epoch information to prefix to saved files
	'''
	file_prefix = summary.data.unique()[0] + '-' + summary.args_group.unique()[0]
	
	return file_prefix

def move_cols(
	df: pd.DataFrame, 
	cols_to_move: List[str] = [], 
	ref_col: str = None, 
	position: str = 'after'
) -> pd.DataFrame:
    '''
    Reorders columns in a dataframe by name and (optionally) relative position to a reference column.
    From https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5
    
    	params:
    		df (pd.DataFrame)	: a dataframe
    		cols_to_move (list)	: a list of column names to be moved
    		ref_col (str)		: the column relative to which the cols to move should be positioned
    		place (str)			: one of 'before', 'after'. whether to place the cols to move before or after the ref col
    	
    	returns:
    		df (pd.DataFrame)	: the dataframe with the columns reordered
    '''
    cols 		= list(df.columns)
    cols_to_move = listify(cols_to_move)
    
    # if the ref col is not provide or is not in the columns, move columns to the front
    index 		= cols.index(ref_col) if ref_col is not None and ref_col in cols else 0
    position 	= 'before' if index == 0 else position
           
    if position == 'after':
        seg1 = cols[:index+1]
        seg2 = cols_to_move
    else:
        seg1 = cols[:index]
        seg2 = cols_to_move + ([ref_col] if ref_col else [])
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])

@crawler(str)
def format_data_for_tokenizer(data: str, mask_token: str, lower: bool) -> str:
	'''
	Format a string for use with a tokenizer
	Recursor means that this applies recursively to any nested data structure, formatting all tokens,
	and outputs data in the same shape as the input
	
		params:
			data (str)			: the data to format for use with a tokenizer
			mask_token (str)	: the tokenizer's mask token
		
		returns:
			the data formatted for use with the tokenizer in string_id
	'''
	return data.lower().replace(mask_token.lower(), mask_token) if lower else data

def format_checkpoint(s: str) -> str:
	'''Formats a string with acheckpoint for plotting/display.'''
	if 'step' in s:
		s = int(re.findall('-step_(.*)k', s)[0])
	else:
		if '1M' in s:
			s = '1M'
		elif '10M' in s:
			s = '10M'
		elif '100M' in s:
			s = '100M'
		elif '1B' in s:
			s = '1000M'
	
	return s