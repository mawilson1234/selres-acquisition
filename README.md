# selres-acquisition

Examining whether pre-trained MultiBERT checkpoints replicate patterns found in children's acquisition of argument structure. Code is modified from the [clay-lab/structural-alternations](https://github.com/clay-lab/structural-alternations) repo.

## Installation

Dependencies are managed using `conda`. To set up the conda environment for the framework, issue the following command from within the `structural-alternations` directory.
```bash
conda env create -f environment.yaml
```
Once the environment has been created, activate with `conda activate selres`.

## Usage

Experiments involve taking an off-the-shelf pre-trained model and examining its logit distributions on masked language modeling (MLM) tasks for a variety of pre-determined tokens or token groups, allowing you to examine things like entropy or token-group confidence in particular positions in testing data for pre-trained BERT models.

The intention behind this is to examine the acquisition path for various kinds of selectional restrictions of checkpoint saved during pre-training of MultiBERT models [(Sellam et al. 2021)](https://arxiv.org/abs/2106.16163), and see how this correlates with pattern observed during human language acquisition.

### Configuration

Configuration is handled using [Hydra](https://github.com/facebookresearch/hydra). Default values are specified in `.yaml` files located in the `conf` directory (and subdirectories). when running from the command line, default values are overridden using `key=value` syntax. Additional explanation of how to flexibly specify parameters using Hydra can be found at [hydra.cc](https://hydra.cc/).

#### Options and defaults

TODO

* `use_gpu (false)`: whether to use GPU support.
* `rerun (false)`: whether to rerun evaluations on directories already containing the expected number of results files.

### Framework Interface

TODO