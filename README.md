# CS232B Midterm Project

## File structure
- `/scripts`: directory containing scripts for finetuning and evaluation
  - `finetune.py`: script for finetuning a LLM
  - `evaluation.py`: script for conducting evaluation automatically
- `/src`: directory containing python modules for running scripts
  - `dataloader.py`: module for representing data as HuggingFace's `Dataset`
  - `events.py`: module for representing a single data samle (i.e. a pair of events)
  - `trainer.py`: module for customizing a weighted trainer
  - `utils`: module containing helper functions
- `/notebooks`: directory containing Jupyter notebooks for experiments and hyperparameter tuning
- `config.yaml`: configuration file for training a LLM

## Usage
Assume the current directory is the project root directory:

**Step 1**: Create a python environment
```
pip install -r requirements.txt
```

**Step 2**: Configure setting for training model
In `config.yaml`, change the field `checkpoint` to desired checkpoint on HuggingFace and according hyperparameters along with the directory to save the fine-tuned models.

> [!NOTE]
> Be careful with the field `decoder_only`, if selected checkpoint is a decoder-only model, it should be python boolean `False` otherwise `True`.

**Step 3**: Finetune a model
```
python /scripts/finetune.py \
> --configFile config.yaml \
> --data ./data
```

**Step 4**: Evaluate the fine-tuned model
```
python /scripts/evaluation.py \
> --model roberta \
> --decoderOnly False \
> --out results.json
```

> [!IMPORTANT]
> The `model` and `decoderOnly` options for the script must align with the values in `config.yaml` otherwise the script would fail to find the directory where contains the model.

**Step 5** (Optional) Hyperparameter tuning
```
jupyter nbconvert \
> --execute notebooks/tuning.ipynb \
> --to notebook \
> --inplace
```

All finetuned models would be saved in the directory `/models`.