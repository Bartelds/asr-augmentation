import omegaconf as oc
import transformers as hft

from helpers import (
    utils,
    w2v2
)

config = oc.OmegaConf.from_cli()

assert '--config' in config.keys(), """\n
    Please supply a base config file, e.g. 'python train.py --config=CONFIG_FILE.yml'.

    You can then over-ride config parameters, e.g. 'python train.py --config=CONFIG_FILE.yml trainargs.learning_rate=1e-5'
"""

config, wandb_run = utils.make_config(config)

utils.announce("Configuring model")

model, processor = w2v2.configure_hf_w2v2_model(config)

datasets = utils.load_datasets(config['data'], processor)

hft.logging.set_verbosity_info()

trainer = w2v2.ReplicationTrainer(
    model=model,
    data_collator=w2v2.DataCollatorCTCWithPadding(processor=processor, padding=True),
    args=hft.TrainingArguments(**config['trainargs']),
    compute_metrics=w2v2.MetricsComputer(config, processor),
    train_dataset=datasets['train'],
    eval_dataset=datasets['eval'],
    tokenizer=processor.feature_extractor
)

utils.announce("Beginning training")

trainer.train()
