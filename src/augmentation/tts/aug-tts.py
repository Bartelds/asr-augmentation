import datasets
import os
import torch
import pandas as pd
import scipy.io.wavfile
from espnet2.bin.tts_inference import Text2Speech
from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser(
    prog='aug-tts'
)

parser.add_argument('--file', default='/gos-demo/train.tsv')

args = parser.parse_args()

data = pd.read_csv(args.file, sep='\t')

data = datasets.Dataset.from_pandas(data)

# load speech model
gos_text2speech = Text2Speech.from_pretrained(
    model_tag="https://huggingface.co/ahnafsamin/FastSpeech2-gronings/resolve/main/tts_train_fastspeech2_raw_char_tacotron_train.loss.ave.zip",
    vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v3",
    device="cuda"
)

def gen_wav(text, outdir):
    with torch.no_grad():
          wav = gos_text2speech(text)['wav']
          scipy.io.wavfile.write(outdir + '-'.join(text.strip().split())[:100] + '.wav', gos_text2speech.fs, wav.view(-1).cpu().numpy())

print(f"Generating wavs...")

outdir = 'data/wav/'
os.makedirs(outdir, exist_ok=True)

for text in tqdm(data['text']):
    gen_wav(text, outdir)
