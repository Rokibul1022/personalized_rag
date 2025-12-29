"""
Fine-tune Qwen or Gemma on Educational RAG Data
Uses LoRA for efficient fine-tuning
"""

import json
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

PROFILES_DIR = Path('personalized_rag/user_profiles')
MODELS_DIR = Path('personalized_rag/local_models')
MODELS_DIR.mkdir(exist_ok=True)

def prepare_training_data():
    """Collect all user interactions for training"""
    training_data = []
    
    # Collect from all user knowledge bases
    for kb_file in PROFILES_DIR.glob('*_knowledge_base.csv'):
        try:
            df = pd.read_csv(kb_file)
            queries = df[df['type'] == 'query']
            
            for _, row in queries.iterrows():
                if pd.notna(row['query']) and pd.notna(row['response']):
                    training_data.append({
                        'instruction': row['query'],
                        'output': row['response'][:500],  # Limit length
                        'topic': row['topic']
                    })
        except Exception as e:
            print(f"Error reading {kb_file}: {e}")
    
    print(f"✅ Collected {len(training_data)} training examples")
    return training_data

def format_prompt(instruction, output):
    """Format data for instruction tuning"""
    return f"""### Instruction:
You are an educational assistant. Answer the following question clearly and concisely.

{instruction}

### Response:
{output}"""

def create_dataset(data):
    """Convert to HuggingFace dataset"""
    formatted_data = []
    for item in data:
        text = format_prompt(item['instruction'], item['output'])
        formatted_data.append({'text': text})
    
    return Dataset.from_list(formatted_data)

def finetune_model(model_name='Qwen/Qwen2.5-3B-Instruct', output_dir='finetuned_model'):
    """Fine-tune model with LoRA"""
    
    print(f"🚀 Starting fine-tuning: {model_name}")
    
    # 1. Prepare data
    print("\n Preparing training data...")
    training_data = prepare_training_data()
    
    if len(training_data) < 10:
        print(" Not enough training data (need at least 10 examples)")
        return
    
    dataset = create_dataset(training_data)
    
    # 2. Load model and tokenizer
    print(f"\n Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # 3. Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print(f" Trainable parameters: {model.print_trainable_parameters()}")
    
    # 4. Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to='none'
    )
    
    # 6. Train
    print("\n🏋️ Training model...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    
    # 7. Save
    print(f"\n Saving to {MODELS_DIR / output_dir}")
    model.save_pretrained(MODELS_DIR / output_dir)
    tokenizer.save_pretrained(MODELS_DIR / output_dir)
    
    print("\n Fine-tuning complete!")
    print(f" Model saved to: {MODELS_DIR / output_dir}")
    print("\n To use: Update local_llm_rag.py to load this model")

def export_for_ollama(model_dir='finetuned_model'):
    """Export fine-tuned model to Ollama format"""
    print("\n Exporting to Ollama format...")
    
    # Create Modelfile
    modelfile = f"""FROM {MODELS_DIR / model_dir}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are an educational assistant specialized in personalized learning."""
    
    modelfile_path = MODELS_DIR / 'Modelfile'
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    
    print(f" Modelfile created: {modelfile_path}")
    print("\n To create Ollama model, run:")
    print(f"   cd {MODELS_DIR}")
    print(f"   ollama create edu-assistant -f Modelfile")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct', 
                       help='Base model to fine-tune')
    parser.add_argument('--output', default='finetuned_qwen', 
                       help='Output directory name')
    parser.add_argument('--export-ollama', action='store_true',
                       help='Export to Ollama format')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎓 Educational RAG - LLM Fine-tuning")
    print("=" * 60)
    
    finetune_model(args.model, args.output)
    
    if args.export_ollama:
        export_for_ollama(args.output)
