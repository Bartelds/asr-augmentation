import jiwer
import os
import torch
import numpy as np
import omegaconf as oc
import pandas as pd
import soundfile as sf
from datasets import Dataset
from transformers import set_seed
from tqdm import tqdm


from helpers import (
    utils,
    w2v2
)

set_seed(4892)
config = oc.OmegaConf.from_cli()

assert '--config' in config.keys(), """\n
    Please supply a base config file, e.g. 'python aug-st.py --config=CONFIG_FILE.yml'.
    You can then over-ride config parameters, e.g. 'python aug-st.py --config=CONFIG_FILE.yml w2v2.model.pretrained_model_name_or_path=your_teacher_model'
"""

config, wandb_run = utils.make_config(config)

utils.announce('Configuring model and reading data')

model, processor = w2v2.configure_hf_w2v2_model(config)
model = model.eval().cuda()

devset_path = os.path.join(config['data'].base_path, config['data'].transcribe_tsv)

print(f"Data to transcribe: {devset_path}")

dev_ds = pd.read_csv(devset_path, sep = '\t')

def _read_audio(path):
    full_path = os.path.join(config['data'].base_path, path)

    data, sr = sf.read(full_path)

    assert sr == 16_000

    return data

dev_ds['audio'] = [ _read_audio(path) for path in tqdm(dev_ds['path'].to_list(), desc='Reading audio data') ]

dev_ds = Dataset.from_pandas(dev_ds[['audio', 'path']])

utils.announce('Evaluating model')

def evaluate(batch):
    inputs = processor(batch['audio'], sampling_rate=16_000, return_tensors='pt', padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to('cuda'), attention_mask=inputs.attention_mask.to('cuda')).logits

    pred_ids = np.argmax(logits.cpu(), axis=-1)
    batch['transcription'] = processor.batch_decode(pred_ids)
    
    return batch

dev_ds = dev_ds.map(evaluate, batched=True, batch_size=8)
dev_ds = dev_ds.to_pandas()
utils.announce('Transcribing')
os.makedirs('./data/transcriptions/', exist_ok=True)
dev_ds[['path', 'transcription']].to_csv('./data/transcriptions/file.tsv', sep = '\t')
