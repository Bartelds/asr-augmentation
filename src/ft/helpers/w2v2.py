import dataclasses
import datasets as hfds
import jiwer
import numpy as np
import pandas as pd
import torch
import transformers as hft
import typing
import wandb
import json

def configure_hf_w2v2_model(config):

    print(f"Loading {config['w2v2']['model']['pretrained_model_name_or_path']} model ...")

    # Set verbosity to error while loading models (skips warnings about loading a not-yet fine-tuned model)
    hft.logging.set_verbosity_error()

    # Re-use the vocab.json from the fine-tuned model instead of re-deriving it from the train/test data

    # !wget https://huggingface.co/facebook/wav2vec2-large-960h/raw/main/vocab.json

    if config['w2v2']['tok']['vocab_file'] is None:
        # Load tokenizer from model if already fine-tuned
        processor = hft.Wav2Vec2Processor.from_pretrained(config['w2v2']['model']['pretrained_model_name_or_path'])

    else:
        # Create a new processor (i.e. fine-tuning for the first time)
        processor = hft.Wav2Vec2Processor(
            tokenizer=hft.Wav2Vec2CTCTokenizer(**(config['w2v2']['tok'] or {})),
            feature_extractor=hft.Wav2Vec2FeatureExtractor(**(config['w2v2']['fext'] or {})),
            **(config['w2v2']['proc'] or {})
        )

    processor.save_pretrained(config['trainargs']['output_dir'])

    model_config = hft.AutoConfig.from_pretrained(config['w2v2']['model']['pretrained_model_name_or_path'])

    # set vocab size
    f = open(config['w2v2']['tok']['vocab_file'])
    vocab = json.load(f)
    vocab_size = max(vocab.values()) + 1
    model_config.vocab_size = vocab_size

    config['w2v2']['model']['pad_token_id'] = processor.tokenizer.pad_token_id
    config['w2v2']['model']['ctc_zero_infinity'] = True

    model_config.update(config['w2v2']['model'])

    model = hft.Wav2Vec2ForCTC.from_pretrained(
        config['w2v2']['model']['pretrained_model_name_or_path'],
        config=model_config
    )

    model.freeze_feature_encoder()

    return model, processor

@dataclasses.dataclass
class DataCollatorCTCWithPadding:

    processor: hft.Wav2Vec2Processor
    padding: typing.Union[bool, str] = True

    def __call__(self, features: typing.List[typing.Dict[str, typing.Union[typing.List[int], torch.Tensor]]]) -> typing.Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

class MetricsComputer:

    def __init__(self, config, processor):

        self.processor = processor
        self.report_to = config['trainargs']['report_to']

        decode_method = config['w2v2']['decode']['method']

        assert decode_method in ['greedy', 'beam_search'], f"\n\tError: Unrecognized decoding method '{decode_method}'"

        if decode_method == 'greedy':

            self.decoder = self.greedy_decoder

        elif decode_method == 'beam_search':

            from torchaudio.models.decoder import ctc_decoder
            from functools import partial

            _decoder = ctc_decoder(
                tokens=list(processor.tokenizer.get_vocab().keys()),
                blank_token=processor.tokenizer.pad_token,
                sil_token=processor.tokenizer.word_delimiter_token,
                unk_word=processor.tokenizer.unk_token,
                **config['w2v2']['decode']['args']
            )

            self.decoder = partial(self.beam_search_decoder, decoder=_decoder)

    def __call__(self, pred):

        labels = self.get_labels(pred)
        preds  = self.decoder(pred)

        wer, cer = self.compute_metrics(labels, preds)

        return { "wer" : wer, "cer" : cer }

    def get_labels(self, pred):
        # Replace data collator padding with tokenizer's padding
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        # Retrieve labels as characters, e.g. 'hello', from label_ids, e.g. [5, 3, 10, 10, 2] (where 5 = 'h')
        label_str = self.processor.tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        return label_str

    def beam_search_decoder(self, pred, decoder):

        pred_logits = torch.tensor(pred.predictions, dtype=torch.float32)

        from tqdm import tqdm
        from joblib import Parallel, delayed

        def logits_to_preds(logits):
            # unsqueeze to make logits to shape (B=1, T, V) expected by decode
            # instead of just (T, V), where B = batch, T = time steps, V = vocab size
            hypotheses = decoder(logits.unsqueeze(0))

            # Subset to get hypotheses for first example (of batch size 1)
            hypotheses = hypotheses[0]

            # Return top hypothesis as a string
            return " ".join(hypotheses[0].words)

        # Decode in parallel
        pred_str = Parallel(n_jobs=-1, verbose=0, prefer="threads")(delayed(logits_to_preds)(l) for l in tqdm(pred_logits, desc="Running beam search decoding ..."))

        return pred_str

    def greedy_decoder(self, pred):

        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = self.processor.batch_decode(pred_ids)

        return pred_str

    def compute_metrics(self, labels, preds):

        scoring_df = pd.DataFrame({"Reference" : labels, "Prediction"  : preds})

        if self.report_to == 'wandb':
            wandb.log({ "asr_out": wandb.Table(data=scoring_df) })

        # Print two newlines first to separate table from progress bar
        print("\n\n")
        print(scoring_df)

        wer = jiwer.wer(labels, preds)
        cer = jiwer.cer(labels, preds)

        return wer, cer

# Adapted from https://discuss.huggingface.co/t/weights-biases-supporting-wave2vec2-finetuning/4839/4
def get_flat_linear_schedule_with_warmup(optimizer, num_warmup_steps:int, num_training_steps:int, last_epoch:int =-1, lr_warmup_pc=0.1, lr_const_pc=0.4):
    
    def lr_lambda(current_step):
        constant_steps = int(num_training_steps * lr_const_pc)
        warmup_steps = int(num_training_steps * lr_warmup_pc)
        
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps+constant_steps:
            return 1
        else:
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (warmup_steps+constant_steps)))
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_flat_cheduler(name = None, optimizer = None, num_warmup_steps = None, num_training_steps = None):
    return get_flat_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

class ReplicationTrainer(hft.Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_flat_scheduler(self, num_training_steps: int):
        self.lr_scheduler = get_flat_cheduler(optimizer = self.optimizer,
                                              num_training_steps=num_training_steps)

    def create_optimizer_and_scheduler(self, num_training_steps):
        self.create_optimizer()
        self.create_flat_scheduler(num_training_steps)
