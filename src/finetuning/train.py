from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import pandas as pd
from datasets import Dataset

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

path_to_dataset = '/kaggle/input/dataset_justdone.csv'

instruction_prompt = "Rewrite the given AI-generated text to make it sound more human-like, improving fluency, coherence, and naturalness while preserving the original meaning."

# Load the dataset from CSV
# dataset = load_dataset("csv", data_files=path_to_dataset)
df = pd.read_csv(path_to_dataset)

df['instruction'] = instruction_prompt
df = df.rename({
    'LLM': 'input',
    'human': 'output'
}, axis=1)

# Convert Pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into train (80%) and validation (20%)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Get train and validation splits
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Print sizes
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return texts

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset =train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_prompts_func,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        # warmup_steps=5,
        # num_train_epochs=1,
        max_steps=20,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=2,
        # save_steps=10,
        eval_strategy="steps",
        eval_steps=0.2,
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        # lr_scheduler_type="linear",
        seed=3407,
        output_dir="just-done-v5",
        report_to = "wandb",
        dataloader_num_workers=4
    )
)

trainer_stats = trainer.train()