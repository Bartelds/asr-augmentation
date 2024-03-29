# Making More of Little Data: Improving Low-Resource Automatic Speech Recognition Using Data Augmentation
Code associated with the paper: Making More of Little Data: Improving Low-Resource Automatic Speech Recognition Using Data Augmentation.

> **Abstract**: The performance of automatic speech recognition (ASR) systems has advanced substantially in recent years, particularly for languages for which a large amount of transcribed speech is available.
Unfortunately, for low-resource languages, such as minority languages, regional languages or dialects, ASR performance generally remains much lower.
In this study, we investigate whether data augmentation techniques could help improve low-resource ASR performance, focusing on four typologically diverse minority languages or language variants (West Germanic: Gronings, West-Frisian; Malayo-Polynesian: Besemah, Nasal).
For all four languages, we examine the use of self-training, where an ASR system trained with the available human-transcribed data is used to generate transcriptions, which are then combined with the original data to train a new ASR system.
For Gronings, for which there was a pre-existing text-to-speech (TTS) system available, we also examined the use of TTS to generate ASR training data from text-only sources.
We find that using a self-training approach consistently yields improved performance (a relative WER reduction up to 20.5% compared to using an ASR system trained on 24 minutes of manually transcribed speech).
The performance gain from TTS augmentation for Gronings was even stronger (up to 25.5% relative reduction in WER compared to a system based on 24 minutes of manually transcribed speech).
In sum, our results show the benefit of using self-training or (if possible) TTS-generated data as an efficient solution to overcome the limitations of data availability for resource-scarce languages in order to improve ASR performance.

<!-- ## Citation

```bibtex
``` -->

## Installation

```bash
git clone https://github.com/Bartelds/asr-augmentation.git
cd asr-augmentation
pip install -r requirements.txt
```

## Data

In this repository, we provide a small demonstration dataset with Gronings in `data/gos-demo`, based on the dataset released by [San et al. (2021)](https://github.com/fauxneticien/qbe-std_feats_eval). This data is also available on the [Hugging Face Hub](https://huggingface.co/datasets/bartelds/gos-demo).

The full datasets used in the experiments for Gronings, Besemah, and Nasal are available on [Zenodo](https://zenodo.org/record/7946870).
The FAME! ASR corpus for West-Frisian can be obtained by emailing the [authors](https://islrn.org/resources/340-994-352-616-4/).

## License

The code and (pre-trained/fine-tuned) models are released as Apache 2.0 license, as indicated in the `LICENSE` file. The Gronings, Nasal, and Besemah ASR datasets are released as CC-BY 4.0.

## Usage

### Pre-training

To continue pre-training on target domain data:

```bash
accelerate config

accelerate launch --mixed_precision fp16 \
src/cpt/run_wav2vec2_pretraining_no_trainer.py \
--dataset_name=google/fleurs \
--dataset_config_names nl_nl \
--dataset_split_names train \
--output_dir=checkpoints \
--max_train_steps=100000 \
--num_warmup_steps=10000 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-5 \
--weight_decay=0.01 \
--max_duration_in_seconds=30.0 \
--min_duration_in_seconds=1.0 \
--model_name_or_path=facebook/wav2vec2-xls-r-300m \
--logging_steps=100 \
--saving_steps=1000 \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--adam_beta1=0.9 \
--adam_beta2=0.98 \
--adam_epsilon=1e-06 \
--gradient_checkpointing \
--wandb_project_name=your-project \
--wandb_run_name=your-run
```

`src/cpt/run_wav2vec2_pretraining_no_trainer.py` was adapted from the code originally developed by the [The HuggingFace Inc. team](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-pretraining) (available under the Apache License, Version 2.0).
In this example, the [fleurs](https://huggingface.co/datasets/google/fleurs/viewer/nl_nl/train) dataset (licensed under CC-BY 4.0) is used as the target domain data.

### Fine-tuning

To fine-tune a [wav2vec 2.0-based model](https://huggingface.co/models?other=wav2vec2):

```bash
python src/ft/train.py --config=src/ft/config.yaml \
    data.base_path=data/gos-demo/ \
    trainargs.output_dir=gos-demo \
    data.train_tsv=train.tsv \
    data.eval_tsv=dev.tsv \
    data.subset_train.mins=24 \
    data.subset_train.seed=4892 \
    trainargs.save_steps=500 \
    trainargs.load_best_model_at_end=False \
    trainargs.per_device_train_batch_size=8 \
    trainargs.per_device_eval_batch_size=8 \
    trainargs.gradient_accumulation_steps=4
```

The parameters as specified in `config.yaml` can be changed using the dot notation for nested keys. For example, the amount of data used for fine-tuning can be set to 48 minutes using: `data.subset_train.mins=48`. More information is available in the [repository of our ComputEL-6 paper](https://github.com/fauxneticien/w2v2-10min-exps).

### Data Augmentation

**Self-training (ST)** To transcribe unlabeled speech recordings using a [wav2vec 2.0-based model](https://huggingface.co/models?other=wav2vec2):
```bash
python src/augmentation/st/aug-st.py --config=src/augmentation/st/config.yaml \
    data.base_path=data/gos-demo \
    data.transcribe_tsv=train.tsv \
    w2v2.model.pretrained_model_name_or_path=bartelds/gos-gpum-cp0_adp0_24m_1e-5_cp-13000
```

**Text-to-speech (TTS)** To generate synthetic speech of Gronings texts using an existing TTS system (e.g., [FastSpeech2-based TTS](https://huggingface.co/spaces/ahnafsamin/GroTTS-FastSpeech2)):
```bash
python src/augmentation/tts/aug-tts.py --file=data/gos-demo/train.tsv
```

Please supply a config file or file with transcriptions.
You can over-ride config parameters using the dot notation mentioned above.
For example: `python aug-st.py --config=CONFIG_FILE.yml w2v2.model.pretrained_model_name_or_path=your_teacher_model`.

### Evaluation

To compute word-error-rates for fine-tuned models:
```bash
python eval.py --config=config.yaml \
    trainargs.output_dir=test \
    env.WANDB_MODE=offline \
    data.base_path=data/gos-demo/ \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=bartelds/gos-gpu1-cp0_adp0_192m_5e-4_cp-12500 \
    trainargs.per_device_train_batch_size=8 \
    trainargs.per_device_eval_batch_size=8 \
    trainargs.gradient_accumulation_steps=4
```

Similarly, you can over-ride config parameters. For example: `python eval.py --config=CONFIG_FILE.yml w2v2.model.pretrained_model_name_or_path=another_w2v2_model`.

## Models

All pre-trained and fine-tuned models are available on the [Hugging Face Hub](https://huggingface.co/bartelds) 🤗 .

## Gronings

### XLS-R continued pre-trained on Gronings speech:
[wav2vec2-xls-r-300m-gos](https://huggingface.co/bartelds/wav2vec2-xls-r-300m-gos)

### Fine-tuned models:

| Model | Min of data | Learning rate | Checkpoint (steps) | CPT | WER test |
|-------|-------------|---------------|--------------------|-----|----------|
[Hugging Face](https://huggingface.co/bartelds/gos-gpu1-cp1_adp0_24m_no_test_5e-5_cp-13000) | 24 | 5e-5 | 13000 | yes |  0.301 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpum-cp0_adp0_24m_1e-5_cp-13000) | 24 | 1e-5 | 13000 | no |  0.332 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu1-cp1_adp0_48m_no_test_1e-5_cp-12000) | 48 | 1e-5 | 12000 | yes |  0.252 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpum-cp0_adp0_48m_1e-4_cp-10000) | 48 | 1e-4 | 10000 | no |  0.252 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu1-cp1_adp0_96m_no_test_1e-4_cp-11500)  | 96 | 1e-4 | 11500 | yes |  0.193 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpum-cp0_adp0_96m_5e-4_cp-11000) | 96 | 5e-4 | 11000 | no |  0.202 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu6-cp1_adp0_192m_no_test_1e-5_cp-12000) | 192 | 1e-5 | 12000 | yes |  0.144 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu1-cp0_adp0_192m_5e-4_cp-12500) | 192 | 5e-4 | 12500 | no |  0.155 |

**Data Augmentation:**

| Model | Min of data | Learning rate | Checkpoint (steps) | CPT | WER test |
|-------|-------------|---------------|--------------------|-----|----------|
[Hugging Face](https://huggingface.co/bartelds/gos-gpu6-cp1_adp0_168m-silver_24-orig_5e-4_cp-12500) | 24 + 168 ST | 5e-4 | 12500 | yes | 0.282 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpum-cp0_adp0_168m-silver_24-orig_1e-5_cp-13000) | 24 + 168 ST | 1e-5 | 13000 | no | 0.286 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu6-cp0_adp0_2x168m-silver_24-orig_1e-5_cp-11000) | 24 + 2 x 168 ST | 1e-5 | 11000 | no | 0.281 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu6-cp0_adp0_4x168m-silver_24-orig_1e-4_cp-12000) | 24 + 4 x 168 ST | 1e-4 | 12000 | no| 0.264 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpum2-cp0_adp0_168-tts_24-orig_1e-4_cp-12500) | 24 + 168 TTS | 1e-4 | 12500 | no | 0.204 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu6-cp0_adp0_2x168m-tts_24-orig_5e-5_cp-11000) | 24 + 2 x 168 TTS | 5e-5 | 11000 | no | 0.209 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu6-cp0_adp0_4x168m-tts_24-orig_5e-4_cp-12500) | 24 + 4 x 168 TTS | 5e-4 | 12500 | no | 0.198 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu6-cp1_adp0_144m-silver_48-orig_5e-4_cp-11000) | 48 + 144 ST | 5e-4 | 11000 | yes | 0.226 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpum-cp0_adp0_144m-silver_48-orig_5e-5_cp-8000) | 48 + 144 ST | 5e-5 | 8000 | no | 0.230 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpu6-cp1_adp0_96m-silver_96-orig_5e-4_cp-10000) | 96 + 96 ST | 5e-4 | 10000 | yes | 0.183 |
[Hugging Face](https://huggingface.co/bartelds/gos-gpum-cp0_adp0_96m-silver_96-orig_1e-4_cp-11000) | 96 + 96 ST | 1e-4 | 11000 | no | 0.183 |

## West-Frisian

### Fine-tuned models:

| Model | Min of data | Learning rate | Checkpoint (steps) | WER test |
|-------|-------------|---------------|--------------------|----------|
[Hugging Face](https://huggingface.co/bartelds/fry-cp0_adp0_24m_5e-4_cp-12000) | 24 | 5e-4 | 12000 | 0.457 |
[Hugging Face](https://huggingface.co/bartelds/fry-cp0_adp0_48m_1e-4_cp-10000) | 48 | 1e-4 | 10000 | 0.382 |
[Hugging Face](https://huggingface.co/bartelds/fry-gpu6-cp0_adp0_96m_5e-4_cp-12500) | 96 | 5e-4 | 12500 | 0.307 |
[Hugging Face](https://huggingface.co/bartelds/fry-cp0_adp0_192m_1e-5_cp-10000) | 192 | 1e-5 | 10000 | 0.261 |

**Data Augmentation:**

| Model | Min of data | Learning rate | Checkpoint (steps) | CPT | WER test |
|-------|-------------|---------------|--------------------|-----|----------|
[Hugging Face](https://huggingface.co/bartelds/fry-gpu6-cp0_adp0_168m-silver_24-orig_5e-5_cp-12500) | 24 + 168 ST | 5e-5 | 12500 | no | 0.428 |
[Hugging Face](https://huggingface.co/bartelds/fry-gpu6-cp0_adp0_144m-silver_48-orig_5e-5_cp-11500) | 48 + 144 ST | 5e-5 | 11500 | no | 0.352 |
[Hugging Face](https://huggingface.co/bartelds/fry-gpu6-cp0_adp0_96m-silver_96-orig_5e-4_cp-13000) | 96 + 96 ST | 5e-4 | 13000 | no | 0.289 |

## Besemah

### Fine-tuned models:

| Model | Min of data | Learning rate | Checkpoint (steps) | WER test |
|-------|-------------|---------------|--------------------|----------|
[Hugging Face](https://huggingface.co/bartelds/besemah-gpu6-cp0_adp0_24m_1e-5_cp-11500) | 24 | 1e-5 | 11500 | 0.517 |
[Hugging Face](https://huggingface.co/bartelds/besemah-gpu6-cp0_adp0_48m_5e-5_cp-11500) | 48 | 5e-5 | 11500 | 0.423 |
[Hugging Face](https://huggingface.co/bartelds/besemah-gpu6-cp0_adp0_96m_1e-5_cp-13000) | 96 | 1e-5 | 13000 | 0.359 |
[Hugging Face](https://huggingface.co/bartelds/besemah-gpu6-cp0_adp0_192m_5e-4_cp-12000) | 192 | 5e-4 | 12000 | 0.316 |

**Data Augmentation:**

| Model | Min of data | Learning rate | Checkpoint (steps) | CPT | WER test |
|-------|-------------|---------------|--------------------|-----|----------|
[Hugging Face](https://huggingface.co/bartelds/besemah-gpu6-cp0_adp0_168m-silver_24-orig_1e-5_cp-13000) | 24 + 168 ST | 1e-5 | 13000 | no | 0.471 |
[Hugging Face](https://huggingface.co/bartelds/besemah-gpu6-cp0_adp0_144m-silver_48-orig_5e-4_cp-13000) | 48 + 144 ST | 5e-4 | 13000 | no | 0.398 |
[Hugging Face](https://huggingface.co/bartelds/besemah-gpu6-cp0_adp0_96m-silver_96-orig_1e-4_cp-13000) | 96 + 96 ST | 1e-4 | 13000 | no | 0.359 |


## Nasal

### Fine-tuned models:

| Model | Min of data | Learning rate | Checkpoint (steps) | WER test |
|-------|-------------|---------------|--------------------|----------|
[Hugging Face](https://huggingface.co/bartelds/nasal-gpu6-cp0_adp0_24m_1e-4_cp-10500) | 24 | 1e-4 | 10500 | 0.591 |
[Hugging Face](https://huggingface.co/bartelds/nasal-gpu6-cp0_adp0_48m_5e-5_cp-13000) | 48 | 5e-5 | 13000 | 0.509 |
[Hugging Face](https://huggingface.co/bartelds/nasal-gpu6-cp0_adp0_96m_1e-5_cp-12500) | 96 | 1e-5 | 12500 | 0.453 |
[Hugging Face](https://huggingface.co/bartelds/nasal-gpu6-cp0_adp0_192m_1e-5_cp-12000) | 192 | 1e-5 | 12000 | 0.413 |

**Data Augmentation:**

| Model | Min of data | Learning rate | Checkpoint (steps) | CPT | WER test |
|-------|-------------|---------------|--------------------|-----|----------|
[Hugging Face](https://huggingface.co/bartelds/nasal-gpu6-cp0_adp0_168m-silver_24-orig_1e-4_cp-12000) | 24 + 168 ST | 1e-4 | 12000 | no | 0.552 |
[Hugging Face](https://huggingface.co/bartelds/nasal-gpu6-cp0_adp0_144m-silver_48-orig_5e-4_cp-11000) | 48 + 144 ST | 5e-4 | 11000 | no | 0.485 |
[Hugging Face](https://huggingface.co/bartelds/nasal-gpu6-cp0_adp0_96m-silver_96-orig_5e-4_cp-13000) | 96 + 96 ST | 5e-4 | 13000 | no | 0.437 |
