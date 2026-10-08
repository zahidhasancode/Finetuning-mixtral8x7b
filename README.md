# Fine-tuning Mixtral 8x7B Instruct with QLoRA

A study notebook that sets up QLoRA fine-tuning (4-bit base model + LoRA adapters) of `mistralai/Mixtral-8x7B-Instruct-v0.1` on 1,000 Dolly examples from `mosaicml/instruct-v3`, using Hugging Face Transformers, PEFT and TRL.

## Status

**Study project, 2024.** One notebook, `MIxtral8x7b_finetuning.ipynb`, written for Google Colab.

What the saved outputs show:
- The dataset was downloaded, filtered and sampled.
- The quantised model was wrapped with LoRA adapters and the trainable-parameter count was printed.

What they do not show:
- **No training run is recorded.** The training, saving and generation cells have no saved output.
- The `TrainingArguments` cell, as saved, does not run: it passes `eval_steps` twice, which Python rejects with `SyntaxError: keyword argument repeated: eval_steps`.
- No adapter or model is included in the repository, and nothing was uploaded (the "Upload model in hugging face" section is empty).

## What it does

1. Loads Mixtral 8x7B Instruct in 4-bit (NF4) with bitsandbytes.
2. Prepares a small instruction dataset (1,000 train / 100 test rows).
3. Formats each row into Mixtral's `[INST] ... [/INST]` chat format.
4. Adds LoRA adapters with PEFT and configures supervised fine-tuning with TRL's `SFTTrainer`.
5. Defines a helper to generate text from the fine-tuned model.

## How it works

### 1. Base model and quantisation (QLoRA)

```python
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

- 4-bit NF4 weights with double quantisation; computation in bfloat16.
- Loaded with `device_map='auto'`, `use_cache=False` and `attn_implementation="flash_attention_2"`.
- The tokenizer uses the EOS token as the padding token, with right-side padding.
- The notebook uses the instruct model, not the base model, because (as its notes say) a base model would need much more data.

### 2. Dataset

- Hugging Face dataset id: **`mosaicml/instruct-v3`** (columns `prompt`, `response`, `source`).
- Full size, from the saved output: 56,167 train / 6,807 test rows.
- Filtered to rows where `source == "dolly_hhrlhf"`: 34,333 train / 4,771 test rows.
- Random sample with `shuffle(seed=42)`: **1,000 train / 100 test** rows.

### 3. Prompt format

`create_prompt` removes the Alpaca-style header from the dataset's `prompt` column and builds:

```
<s>[INST]You are Helpful Assistant.
{dataset response}[/INST]{dataset instruction}</s>
```

Note the direction: the dataset's **response** goes inside `[INST]`, and the original **instruction** is the target the model learns to write. So, as written, the notebook trains the model to produce an instruction from a given answer (reverse instruction generation). The saved output of `create_prompt` on the first row shows this. The test prompt at the end matches that goal ("create an instruction that could have been used to generate the response").

### 4. LoRA configuration (PEFT)

```python
LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", "lm_head"],
    task_type="CAUSAL_LM",
)
```

- The model goes through `prepare_model_for_kbit_training` and then `get_peft_model`.
- In Mixtral, the expert feed-forward layers are named `w1`, `w2`, `w3` (the notebook has them commented out), so `gate_proj`, `up_proj` and `down_proj` match nothing. The adapters land on the attention projections (`q/k/v/o_proj`) in all 32 layers and on `lm_head`. The experts are not adapted.
- Saved output of `print_trainable_parameters`:
  `trainable params: 56836096 || all params: 23539437568 || trainable%: 0.24145052674182907`.
  This matches rank-64 adapters on `q/k/v/o_proj` × 32 layers plus `lm_head`. The "all params" figure is lower than Mixtral's published ~46.7B parameters because bitsandbytes packs two 4-bit weights into one stored element.

### 5. Training arguments

| Setting | Value in the code |
|---|---|
| `output_dir` | `./Mixtral_new_model` |
| `optim` | `paged_adamw_8bit` |
| `max_steps` | 1000 (epochs line commented out) |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 2 |
| `per_device_eval_batch_size` | 8 |
| `learning_rate` | 2.5e-5 (the `lr_scheduler_type='cosine'` line is commented out, so the default linear schedule applies) |
| `warmup_steps` | 0.03 |
| `logging_steps` | 10 |
| `evaluation_strategy` | `"steps"`, with `eval_steps` given twice (50 and 10) |
| `save_strategy` | `"epoch"` |
| `bf16` | True |

### 6. Trainer

`SFTTrainer` with `max_seq_length=1024`, `packing=True`, `formatting_func=create_prompt`, the LoRA config, and the 1,000 / 100 row splits. Then `trainer.train()` and `trainer.save_model("Mixtral_instruct_v2")`.

### 7. Generation

`generate_response` tokenizes a prompt, calls `model.generate(max_new_tokens=512, do_sample=True)` and strips the prompt from the decoded text.

## Setup and how to run

**GPU.** This needs a large GPU.
- Mixtral 8x7B in 4-bit still needs roughly 24 GB or more of GPU memory for the weights alone, before activations and optimizer state.
- `attn_implementation="flash_attention_2"` needs the `flash-attn` package and an NVIDIA Ampere-or-newer GPU (for example A100, H100, RTX 30/40 series).
- The notebook metadata names a Colab T4 (16 GB, Turing). That GPU is too small and does not support FlashAttention 2.
- The cell that counts GPUs printed `8`, so the saved outputs came from a session with 8 GPUs visible.

**Steps:**
1. `pip install -r requirements.txt` (the notebook's first cell installs the same packages with `pip install -U`). Install `flash-attn` separately if your GPU supports it; otherwise remove the `attn_implementation` argument.
2. If Hugging Face asks for authentication when downloading `mistralai/Mixtral-8x7B-Instruct-v0.1`, log in with `huggingface-cli login` (or a Colab secret named `HF_TOKEN`). Never paste a token into the notebook.
3. Fix the `TrainingArguments` cell before running it: keep one `eval_steps` value, and use `warmup_ratio=0.03` if a 3% warm-up was intended (`warmup_steps` expects a whole number of steps).
4. Run the cells in order.

**Library versions.** The notebook installed the latest releases in mid-2024 and did not record versions, so `requirements.txt` is unpinned. Later releases renamed some arguments used here (for example TRL moved `max_seq_length` and `packing` into `SFTConfig`, and Transformers renamed `evaluation_strategy` to `eval_strategy`). With current versions you will need to adapt those cells.

## Results

**No training results are saved in the notebook outputs.** There is no training loss, evaluation loss or generated text.

The only measurements in the saved outputs are the dataset sizes (above) and the trainable-parameter count: 56,836,096 trainable parameters, 0.24% of the stored parameter count.

## Limitations

- No training run is recorded in the saved notebook, and the training-arguments cell has a syntax error as saved.
- The prompt format swaps instruction and response (see [Prompt format](#3-prompt-format)). This is fine if the goal is instruction generation, but the notebook does not say so.
- The test prompt at the end uses `[\INST]` instead of `[/INST]`, so it does not match the training format.
- `max_steps=1000` with an effective batch of 4 packed sequences per device is likely several passes over a 1,000-example training set. The notebook's own note warns that too many steps will over-fit.
- The LoRA target list includes names that do not exist in Mixtral, so the mixture-of-experts layers are not fine-tuned.
- There is no evaluation beyond the trainer's loss on 100 test rows, and no comparison with the untuned model.

## Tech used

Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), bitsandbytes (4-bit NF4), TRL (`SFTTrainer`), Datasets, Accelerate, FlashAttention 2, Google Colab.

Model: `mistralai/Mixtral-8x7B-Instruct-v0.1`. Dataset: `mosaicml/instruct-v3` (filtered to `dolly_hhrlhf`).
