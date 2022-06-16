# selres_plots.py
#
# plotting functions for selres.py
import re
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import seaborn as sns

from . import selres_utils
from typing import *

def get_plot_title(
	df: pd.DataFrame,
	metric: str = ''
) -> str:
	'''
	Get a plot title for selres plots
	
		params:
			model_name	: a dataframe containing information about the experiment
			metric (str): the metric for which a plot title is being created
		
		returns:
			title (str)	: a plot title with information from df and metric
	'''
	title += f'{model_name} {metric}\n'
	
	return title

# different plot types
def plot_learning_curves(summary: pd.DataFrame) -> None:
	'''Wrapper to create various plots from a single call when a summary df is passed.'''
	with PdfPages(f'{selres_utils.get_file_prefix(summary)}-curves.pdf') as pdf:
		for val in ['odds_ratio', 'gf_ratio_conf', 'entropy', 'token_accuracy', 'grammatical_function_accuracy']:
			if val in summary.columns:
				grid = create_curve(summary=summary, val=val)
				pdf.savefig(bbox_inches='tight')
				plt.close('all')
				del grid
				
				if summary.model_base.unique()[0] == 'multiberts':
					grid = create_curve(summary=summary[summary.checkpoint <= 200], val=val)
					pdf.savefig(bbox_inches='tight')
					plt.close('all')
					del grid

def create_curve(summary: pd.DataFrame, val: str) -> sns.axisgrid.FacetGrid:
	'''
	Plots a curve of mean odds ratios per group over time.
	
		params:
			summary (pd.DataFrame): a summary containing odds ratios information to plot
			val (str)			  : the name of the column containing the data to plot
	'''
	summary = summary.copy(deep=True)
	summary = summary.assign(verb_profile = summary.verb_profile.astype(str))
	col = 'gf_ratio_name'
	if val in ['gf_ratio_conf', 'grammatical_function_accuracy', 'entropy']:
		summary = summary.drop(['odds_ratio', 'token_id', 'token'], axis=1).drop_duplicates(ignore_index=True).reset_index(drop=True)
		if val in ['entropy']:
			summary = summary.drop(['gf_ratio_name', 'linear_ratio_name', 'gf_ratio_conf'], axis=1).drop_duplicates(ignore_index=True).reset_index(drop=True)
			col = 'position'
	
	grid = sns.relplot(
		data=summary,
		x='checkpoint',
		y=val,
		hue='verb_profile',
		kind='line',
		col=col,
		row='voice',
		row_order=['active', 'passive'],
		ci=68,
	)
	
	grid.legend.set_title(grid.legend.get_title().get_text().replace('_', ' '))
	
	for text in grid.legend.get_texts():
		text.set_text(text.get_text().replace('_', ' '))
	
	axes = grid.axes
	for r, _ in enumerate(axes):
		for c, _ in enumerate(axes[r]):
			title = axes[r][c].get_title().replace(f'{col} = ', '')
			title = re.sub(r'(.*) \| (.*)', '\\2, \\1', title).replace('voice = ', '')
			title = re.sub(r'\[|\]', '', title)
			if val == 'odds_ratio':
				title = f'Confidence of {re.findall("(.*)/", title)[0]} arguments in {re.findall("(.*),", title)[0]} position, {re.findall(", (.*)", title)[0]}'
			elif val == 'gf_ratio_conf':
				title = f'Confidence of {re.findall("(.*),", title)[0]} arguments in {re.findall("(.*)/", title)[0]} position, {re.findall(", (.*)", title)[0]}'
			elif val == 'entropy':
				title = f'Entropy of {re.findall("(.*),", title)[0]} position, {re.findall(", (.*)", title)[0]}'
			elif val == 'token_accuracy':
				title = f'Accuracy of {re.findall("(.*)/", title)[0]} arguments in {re.findall("(.*),", title)[0]} position, {re.findall(", (.*)", title)[0]}'
			elif val == 'grammatical_function_accuracy':
				title = f'Accuracy of {re.findall("(.*),", title)[0]} arguments in {re.findall("(.*)/", title)[0]} position, {re.findall(", (.*)", title)[0]}'
			
			axes[r][c].set_title(title, fontsize=10)
			axes[r][c].set_xlabel(axes[r][c].get_xlabel().replace('_', ' '))
			axes[r][c].set_ylabel(axes[r][c].get_ylabel().replace('_', ' '))
			if val not in ['entropy', 'token_accuracy', 'grammatical_function_accuracy']:
				axes[r][c].plot(axes[r][c].get_xlim(), (0,0), linestyle='--', color='k', scalex=False, alpha=0.3, zorder=0)
			
			if all([isinstance(cp,int) for cp in summary.checkpoint.unique()]):
				axes[r][c].set_xticks([t for t in axes[r][c].get_xticks() if int(t) in summary.checkpoint.unique()])
	
	plt.gcf().suptitle(
		summary.string_id.unique()[0].replace('google/', '').replace('-seed_', ' ').replace('nyu-mll/', '').replace('-base-', '_').replace('roberta', 'miniberta').re + 
		', data: ' +
		summary.data.unique()[0].replace('_', ' ') +
		', args: ' +
		summary.args_group.unique()[0].replace('_', ' ')
	)
	plt.subplots_adjust(top=0.925, hspace=0.15)
	
	return grid
