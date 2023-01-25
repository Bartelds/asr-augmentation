import datasets as hfds
import jiwer
import omegaconf as oc
import pandas as pd
import soundfile as sf
import os
import wandb

from tqdm import tqdm

# Taken from https://www.geeksforgeeks.org/print-colors-python-terminal/
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))

def announce(announcement):
    total_width = os.get_terminal_size().columns
    pad_length  = int(total_width/2 - len(announcement)/2 - 1)

    print(f"{'-' * pad_length} {announcement} {'-' * pad_length}")

def make_config(config):

    # Overwrite config vars with anything supplied in the command line
    config = oc.OmegaConf.merge(
        oc.OmegaConf.load(config['--config']),
        oc.OmegaConf.from_cli()
    )

    flat_args_long = pd.json_normalize(oc.OmegaConf.to_container(config), sep=".").melt(var_name='argument')
    missing_args   = flat_args_long.query("value == '???'")

    assert len(missing_args) == 0, f"""
    
    The following required arguments are missing:
    
        {','.join(missing_args['argument'].to_list())}

    """

    announce("Configuring environment")

    # Set environment variables
    for key, value in config['env'].items():

        if key == 'CUDA_VISIBLE_DEVICES':
            # OmegaConf will coerce number-like values into integers
            # but CUDA_VISIBLE_DEVICES should be a (comma-seperated) string
            value = str(value)

        os.environ[key] = value

    if not 'wandb' in config.keys():

        return config, None

    else:

        run = wandb.init(allow_val_change=True, settings=wandb.Settings(code_dir="."), **config['wandb'])

        if config.get("--run_name"):
            # Interpolate 'lr={tranargs[learning_rate]}' to 'lr=0.0001', where config['tranargs']['learning_rate'] = 0.0001
            run.name = config["--run_name"].format(**config)

        # Log hyper-parameters not automatically tracked by wandb
        untracked_args = flat_args_long[ ~flat_args_long.argument.str.contains("w2v2|trainargs|wandb|--", regex=True) ]
        # Convert to flat dict, e.g. { 'data.base_path' : '/path/to/the/data' }
        untracked_args = dict([ (d['argument'], d['value']) for d in untracked_args.to_dict(orient='records') ])

        wandb.config.update(untracked_args, allow_val_change=True)

        config['trainargs']['report_to'] = "wandb"

        return config, run

def load_datasets(data_config, processor):

    announce("Loading data ...")

    def _tsv2ds(tsv_file):

        tsv_path = os.path.join(data_config['base_path'], data_config[tsv_file])

        print(f"Reading split from {tsv_path}")

        df = pd.read_csv(tsv_path, sep='\t')

        for c in ['path_col', 'text_col']:
            col_name = data_config[c]

            assert col_name in df.columns, f"\n\n\tDataset {tsv_path} is missing '{col_name}' column\n"

        # Normalize column names
        df = df.rename(columns = {
            data_config['path_col'] : 'path',
            data_config['text_col'] : 'text'
        })

        def _read_audio(path):
            full_path = os.path.join(data_config['base_path'], path)

            data, sr = sf.read(full_path)

            assert sr == 16_000

            return data

        df['audio'] = [ _read_audio(path) for path in tqdm(df['path'].to_list(), desc="Reading audio data") ]

        if 'subset_train' in data_config and tsv_file == 'train_tsv':

            df = df.sample(frac=1, random_state=data_config['subset_train']['seed']).copy().reset_index(drop=True)
            df = df[ df['audio'].apply(lambda s: len(s)/16_000).cumsum() <= (60 * data_config['subset_train']['mins']) ].copy().reset_index(drop=True)

            prYellow(f"Subsetted training data as specified: {data_config['subset_train']['mins']} minutes, random seed {data_config['subset_train']['seed']}. Rows kept: {len(df)}")
        
        # see files in subset
        print("Files in training subset:")
        for f in df['path'].to_list():
            print(f)

        dataset = hfds.Dataset.from_pandas(df[['audio', 'text']])

        return dataset

    datasets = hfds.DatasetDict({
        'train' : _tsv2ds('train_tsv'),
        'eval' : _tsv2ds('eval_tsv')
    })

    def _to_inputs_and_labels(batch):
        batch["input_values"] = processor(batch["audio"], sampling_rate=16000).input_values[0]
    
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
        return batch

    announce("Preparing input features and labels ...")

    datasets = datasets.map(_to_inputs_and_labels, remove_columns=['audio', 'text'])

    return datasets
