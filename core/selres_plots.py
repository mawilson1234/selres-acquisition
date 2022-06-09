# selres_plots.py
#
# plotting functions for selres.py
import logging

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
from matplotlib import pyplot as plt

from typing import *
from contextlib import suppress

log = logging.getLogger(__name__)

# main plotting functions
def scatterplot(
	data: pd.DataFrame, 
	x: pd.DataFrame, 
	y: pd.DataFrame, 
	val: str, 
	hue: str, 
	ax: matplotlib.axes.Axes, 
	sem: str = None,
	text: str = None, 
	text_color: str = None,
	text_size: str = None, 
	center_at_origin: bool = False,
	aspect_ratio: str = None, 
	comparison_line: bool = False,
	legend_title: str = None, 
	legend_labels: Dict = None, 
	diffs_plot: bool = False,
	marginal_means: List[str] = None, 
	xlabel: str = None, 
	ylabel: str = None,
	plot_kwargs: Dict = {}, 
	line_kwargs: Dict = {}, 
	text_kwargs: Dict = {}, 
	label_kwargs: Dict = {},
) -> matplotlib.axes.Axes:
	'''
	The main function used for creating scatterplots for selres objects
	
		params:
			data (pd.DataFrame) 		: a dataframe containing information about the data to plot
			x (pd.DataFrame)			: a dataframe containing information and values to plot on the x-axis.
										  Note that passing a column name will not work, unlike in matplotlib or seaborn
			y (pd.DataFrame)			: a dataframe containing information and values to plot on the y-axis.
										  Note that passing a column name will not work, unlike in matplotlib or seaborn
			val (str)					: the measure to plot on the x and y axes. only one measure may be plotted (i.e., x and y show the same kind of data)
			hue (str)					: a string indicating a column in the dataframes to use for group colors
			ax (matplotlib.axes.Axes)	: an object to plot on
			sem (str)					: name of column containing standard error values in x and y
			center_at_origin (bool)		: whether to center the plot at the origin
			aspect_ratio (str)			: whether to adjust the aspect ratio of the plot. should be one of 'square', 'eq_square'
										  'eq_square' produces a plot with equal-size x- and y-axes with equal limits
										  'square' produces a plot with visually equal-size x- and y-axes, but with possible different ranges on x and y axes
			legend_title (str)			: what to title the plot legend (if it exists)
			legend_labels (dict)		: a dictionary mapping the values in the dataframes' hue columns to display labels
			marginal_means (list)		: a list of column names, each separate grouping of which to add ticks for the group mean and standard error on the plot's margins for
			xlabel (str)				: the label of the x-axis
			ylabel (str)				: the label of the y-axis
			plot_kwargs (dict)			: arguments passed to sns scatterplot
			line_kwargs (dict)			: arguments passed to ax.plot (used to draw comparison lines)
			label_kwargs (dict)			: arguments passed to ax.set_xlabel and ax.set_ylabel (when xlabel and ylabel are provided)
		
		returns:
			ax (matplotlib.axes.Axes)	: the ax object after the plot has been created
	'''		
	data = data.copy()
	x = x.copy()
	y = y.copy()
	
	# seaborn can't plot tensors, so make sure everything is float
	for col in [val, sem]:
		for df in [x, y]:
			if col in df:
				df[col] = df[col].astype(float)
	
	sns.scatterplot(data=data, x=x[val], y=y[val], hue=hue, ax=ax, **plot_kwargs)
	
	(xllim, xulim), (yllim, yulim) = get_set_plot_limits(ax, center_at_origin, aspect_ratio, diffs_plot)
	
	if comparison_line:
		ax.plot((xllim, xulim), (0,0), linestyle='--', color='k', scalex=False, scaley=False, zorder=0, alpha=.3, **line_kwargs)
	
	if aspect_ratio is not None and 'square' in aspect_ratio:
		ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')
	
	set_legend_title(ax, legend_title)
	
	set_legend_labels(ax, legend_labels)
	
	if marginal_means:
		colors = [collection._original_edgecolor for collection in ax.collections[1:]]
		
		# if we've added the errorbars, it repeats the original colors twice (once for each axis the error bar is shown on) 
		# we just want to pass the unique colors
		colors = sorted(set(colors), key=colors.index)
		add_marginal_mean_ticks_to_plot(x=x, y=y, val=val, ax=ax, groups=marginal_means, colors=colors)
	
	if xlabel is not None:
		ax.set_xlabel(xlabel, **label_kwargs)
	
	if ylabel is not None:
		ax.set_ylabel(ylabel, **label_kwargs)
	
	return ax

def add_marginal_mean_ticks_to_plot(
	x: pd.DataFrame, 
	y: pd.DataFrame, 
	val: str, 
	ax: matplotlib.axes.Axes,
	colors: List[Tuple[float]] = None,
	groups: List[str] = None,
) -> matplotlib.axes.Axes:
	'''
	Adds ticks to the margins of the plot for means and standard devations for groups
	
		params:
			x (pd.DataFrame)			: the data plotted on the x-axis
			y (pd.DataFrame)			: the data plotted on the y-axis
			val (str)					: which value to plot group means for
			ax (matplotlib.axes.Axes)	: the ax object to add mean ticks to
			colors (list)				: a list of colors (expressed as float tuples) corresponding to groups grouped by color in the plot
			groups (list)				: a list of columns in x and y. for each combination of unique values in all columns, a separate mean tick and se will be adde
		
		returns:
			ax (matplotlib.axes.Axes)	: the ax object with the mean and standard error ticks added to the margins
	'''
	# here we add ticks to show the mean and standard errors along each axis
	groups = [groups] if isinstance(groups, str) else groups
	
	if colors is None:
		# if no colors are passed, default to black
		colors = [(0.,0.,0.)]
	
	if groups is not None and all(all(group in data.columns for group in groups) for data in [x, y]):
		# this makes sure that the colors used in the plot and the colors used for the groups in the plot and the colors used for marginal mean ticks match
		groups 		= sorted(groups, key = lambda group: x[group].unique().size == len(colors))
		
		# repeat the colors as needed for each additional group
		for group in groups[:-1]:
			colors  *= x[group].unique().size
		
		group_means = [data.groupby(groups)[val].agg({'mean', 'sem'}) for data in [x,y]]
	else:
		group_means = [data[val].agg('mean', 'sem') for data in [x,y]]
	
	xllim, xulim 	= ax.get_xlim()
	yllim, yulim 	= ax.get_ylim()
	xtick_range 	= (xulim - xllim)/30
	ytick_range 	= (yulim - yllim)/30
	
	line_kwargs 	= dict(linestyle='-', zorder=0, scalex=False, scaley=False, alpha=.3)
	
	for axis, group_mean, llim, tick_range in zip(['x', 'y'], group_means, [yllim, xllim], [xtick_range, ytick_range]):
		for (groupname, group), color in zip(group_mean.groupby(groups), colors):
			
			line_kwargs.update(dict(color=color))
			
			x_loc 		= (group.loc[groupname, 'mean'], group.loc[groupname, 'mean'])
			y_loc 		= (llim, llim + tick_range)
			xsem_loc 	= (x_loc - group.loc[groupname, 'sem'], x_loc + group.loc[groupname, 'sem'])
			ysem_loc 	= (llim + tick_range/2, llim + tick_range/2)
			
			if axis == 'y':
				x_loc, y_loc = y_loc, x_loc
				xsem_loc, ysem_loc = ysem_loc, xsem_loc
			
			ax.plot(x_loc, y_loc, **line_kwargs)
			ax.plot(xsem_loc, ysem_loc, linewidth=0.75, **line_kwargs)
	
	return ax

# setters
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

def get_set_plot_limits(
	ax: matplotlib.axes.Axes, 
	center_at_origin: bool = False,
	aspect_ratio: str = '', 
	diffs_plot: bool = False
) -> Tuple:
	'''
	Sets and returns the plot limits for the appropriate plot type
	
		params:
			ax (matplotlib.axes.Axes)	: the ax object to get/set limits for
			center_at_origin (bool)		: whether to center the plot at the origin
			aspect_ratio (str)			: what aspect ratio to use. currently, only 'eq_square' is implemented
										  'eq_square' produces a plot with x and y axis that cover the same range
			diffs_plot (str)			: whether this is a plot where y = y - x
		
		returns:
		 	limits (tuple)				: a tuple with ((x lower lim, x upper lim), (y lower lim), (y upper lim))
		 								  according to the passed parameters
	'''
	if center_at_origin:
		xulim = max([*np.abs(ax.get_xlim()), *np.abs(ax.get_ylim())])
		xulim += (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		yulim = xulim
		xllim, yllim = -xulim, -yulim
	elif aspect_ratio == 'eq_square' and not diffs_plot:
		xulim = max([*ax.get_xlim(), *ax.get_ylim()])
		xulim += (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		xllim = min([*ax.get_xlim(), *ax.get_ylim()])
		xllim -= (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		yulim = xulim
		yllim = xllim
	else:
		xllim, xulim = ax.get_xlim()
		xllim -= (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		xulim += (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		yllim, yulim = ax.get_ylim()
		yllim -= (ax.get_ylim()[1] - ax.get_ylim()[0])/32
		yulim += (ax.get_ylim()[1] - ax.get_ylim()[0])/32
	
	if diffs_plot:
		yulim = max(np.abs(ax.get_ylim()))
		yulim += (ax.get_ylim()[1] - ax.get_ylim()[0])/32
		yllim = -yulim
	
	ax.set_xlim([xllim, xulim])
	ax.set_ylim([yllim, yulim])
	
	return (xllim, xulim), (yllim, yulim)

def set_legend_title(
	ax: matplotlib.axes.Axes,
	legend_title: str = None
) -> None:
	'''
	Sets the plot legend title
	
		params:
			ax (matplotlib.axes.Axes)	: the plot to set the legend title for
			legend_title (str)			: what to set the legend title to
	'''
	if legend_title is not None:
		if legend_title != '':
			try:
				ax.get_legend().set_title(legend_title)
			except AttributeError:
				log.warning('A legend title was provided but no legend exists.')
		
		elif legend_title == '':
			# if the legend title is '', we want to delete the whole thing
			with suppress(AttributeError):
				# this ensures the title exists
				# if it doesn't, an error is thrown and we'll exit without overwriting the wrong thing
				_ = ax.get_legend().get_title()
				handles, labels = ax.get_legend_handles_labels()
				ax.legend(handles=handles, labels=labels)

def set_legend_labels(
	ax: matplotlib.axes.Axes,
	legend_labels: Dict = None
) -> None:
	'''
	Set the plots legend labels
	
		params:
			ax (matplotlib.axes.Axes)	: the plot to set legend labels for
			legend_labels (Dict)		: a dict mapping the default legend labels (taken from colnames in the data)
										  to the desired display labels
	'''
	if legend_labels is not None:
		for text in ax.get_legend().get_texts():
			with suppress(KeyError, AttributeError):
				text.set_text(legend_labels[text.get_text()])

# different plot types
def create_plots(*args, **kwargs) -> None:
	'''Wrapper to create various plots from a single call when a summary df is passed.'''
	create_odds_ratios_plot(*args, **kwargs)
	create_group_conf_plot(*args, **kwargs)
	create_entropy_plot(*args, **kwargs)
	# we want these to be line plots over time for the various model checkpoints. probably get means for each checkpoint
	# and then plot? linestyle for sentence types and colors for verb classes