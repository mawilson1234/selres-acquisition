# eval.py
# 
# Application entry point for evaluating masked language models.
import hydra

from core.selres import SelectionalRestrictionEvaluator
from omegaconf import OmegaConf, DictConfig

OmegaConf.register_new_resolver(
	'model_name', 
	lambda string_id: string_id.replace('google/', '').replace('-seed', '').replace('nyu-mll/', '').replace('-base-', '_').replace('roberta', 'miniberta')
)

OmegaConf.register_new_resolver(
	'which_args',
	lambda string_id, which_args: which_args if not which_args == 'model' else string_id.replace('google/', '').replace('-seed', '').replace('nyu-mll/', '').replace('-base-', '_')
)

@hydra.main(config_path='conf', config_name='selres')
def evaluate(cfg: DictConfig) -> None:
	'''
	Evaluates model checkpoints according to the passed config.
	
		params:
			cfg (DictConfig): a DictConfig specifying evaluation parameters.
							  Explanation and defaults can be found in ./conf/eval.yaml.
	'''
	print(OmegaConf.to_yaml(cfg, resolve=True))
	selres = SelectionalRestrictionEvaluator(cfg)
	selres.evaluate()

if __name__ == '__main__':
	
	evaluate()
