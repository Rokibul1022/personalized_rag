"""
Auto-Learning System: Model learns from every query automatically
Integrates continuous fine-tuning with knowledge base growth
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import threading
import time

PROFILES_DIR = Path('personalized_rag/user_profiles')
MODELS_DIR = Path('personalized_rag/local_models')
MODELS_DIR.mkdir(exist_ok=True)

class AutoLearner:
    def __init__(self, finetune_every=20, model='Qwen/Qwen2.5-3B-Instruct'):
        self.finetune_every = finetune_every
        self.model = model
        self.checkpoint_file = MODELS_DIR / 'auto_learning.json'
        self.load_state()
        
    def load_state(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                self.last_count = state.get('last_count', 0)
                self.total_finetunings = state.get('total_finetunings', 0)
        else:
            self.last_count = 0
            self.total_finetunings = 0
    
    def save_state(self):
        state = {
            'last_count': self.last_count,
            'total_finetunings': self.total_finetunings,
            'last_finetune': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_kb_count(self):
        count = 0
        for kb_file in PROFILES_DIR.glob('*_knowledge_base.csv'):
            try:
                df = pd.read_csv(kb_file)
                count += len(df[df['type'] == 'query'])
            except:
                pass
        return count
    
    def should_finetune(self):
        current = self.get_kb_count()
        new_queries = current - self.last_count
        return new_queries >= self.finetune_every
    
    def trigger_finetune(self):
        print(f"\n🎓 AUTO-LEARNING: Fine-tuning after {self.finetune_every} new queries")
        
        result = subprocess.run([
            'python', 'finetune_llm.py',
            '--model', self.model,
            '--output', f'auto_learned_{self.total_finetunings}'
        ], capture_output=True)
        
        if result.returncode == 0:
            self.last_count = self.get_kb_count()
            self.total_finetunings += 1
            self.save_state()
            print(f"✅ Auto-learning complete! Total finetunings: {self.total_finetunings}")
        else:
            print(f"❌ Auto-learning failed: {result.stderr.decode()}")
    
    def check_and_learn(self):
        if self.should_finetune():
            threading.Thread(target=self.trigger_finetune, daemon=True).start()

# Global auto-learner instance
auto_learner = None

def init_auto_learning(finetune_every=20):
    global auto_learner
    auto_learner = AutoLearner(finetune_every=finetune_every)
    print(f"✅ Auto-learning enabled: Will finetune every {finetune_every} queries")
    return auto_learner

def on_query_saved():
    """Call this after saving each query to KB"""
    if auto_learner:
        auto_learner.check_and_learn()
