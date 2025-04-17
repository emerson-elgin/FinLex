import os
import json
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging as transformers_logging
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_info()

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LLAMA_TRAIN_PATH = os.path.join(DATA_DIR, "llama_train.json")
LLAMA_EVAL_PATH = os.path.join(DATA_DIR, "llama_eval.json")
OUTPUT_DIR = os.path.join(MODELS_DIR, "llama-2-7b-finsentiment")

# Model Configuration 
MODEL_ID = "meta-llama/Llama-2-7b-hf"  # You can also use 'meta-llama/Llama-2-7b-chat-hf' for the chat variant
DEVICE_MAP = "auto"

def create_dataset_from_json(json_file):
    with open(json_file, 'r') as f:
        examples = json.load(f)
    
    dataset = {
        "text": [example["text"] for example in examples]
    }
    return dataset

def train():
    """
    Fine-tune Llama 2.7 for financial sentiment analysis
    """
    # Make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load training and evaluation datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset("json", data_files=LLAMA_TRAIN_PATH, split="train")
    eval_dataset = load_dataset("json", data_files=LLAMA_EVAL_PATH, split="train")
    
    # Configure quantization for efficient training
    logger.info("Configuring model for training...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model
    logger.info(f"Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP,
        trust_remote_code=True,
        token=True  # You'll need a Hugging Face token with access to Llama models
    )
    model.config.use_cache = False  # Recommended for training
    model.config.pretraining_tp = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for training
    logger.info("Preparing model for PEFT/LoRA training")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA adapter
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    # Apply PEFT/LoRA configuration to model
    model = get_peft_model(model, peft_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        eval_steps=0.2,  # Evaluate every 20% of training
        save_strategy="steps",
        save_steps=0.2,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb",  # Optional: comment out if not using W&B
        gradient_checkpointing=True,
        push_to_hub=False,
    )
    
    # Initialize SFT trainer
    logger.info("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=512,
        dataset_text_field="text",
    )
    
    # Train and save the model
    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info(f"Training complete. Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    # Test the trained model with a sample
    logger.info("Testing model with a sample...")
    sample_text = "Apple's stock has risen 10% following better-than-expected earnings, with analysts predicting continued growth in the upcoming quarters."
    pipe = pipeline(task="text-generation", 
                  model=model, 
                  tokenizer=tokenizer, 
                  max_length=200)
    result = pipe(f"<s>[INST] Analyze the sentiment of the following financial text:\n\n{sample_text}\n\nReturn only one of these sentiment labels: positive, negative, or neutral. [/INST]")
    logger.info(f"Sample prediction: {result[0]['generated_text']}")
    
    logger.info("Done! Model trained and saved successfully.")
    return None

if __name__ == "__main__":
    train()
