"""
Continuous Learning System with RLHF-like Feedback
Automatically fine-tunes model based on user feedback
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import time

PROFILES_DIR = Path('personalized_rag/user_profiles')
MODELS_DIR = Path('personalized_rag/local_models')

class ContinuousLearner:
    def __init__(self, threshold=50, check_interval=3600):
        """
        threshold: Number of new good interactions before auto-finetune
        check_interval: Seconds between checks (default 1 hour)
        """
        self.threshold = threshold
        self.check_interval = check_interval
        self.last_finetune_count = self.get_current_count()
        self.running = False
        
    def get_current_count(self):
        """Count total good interactions"""
        count = 0
        for kb_file in PROFILES_DIR.glob('*_knowledge_base.csv'):
            try:
                df = pd.read_csv(kb_file)
                # Count queries (all are considered good if saved)
                count += len(df[df['type'] == 'query'])
            except:
                pass
        return count
    
    def get_feedback_data(self):
        """Get interactions with explicit good feedback"""
        feedback_file = Path('user_interactions.json')
        if not feedback_file.exists():
            return []
        
        with open(feedback_file, 'r') as f:
            interactions = json.load(f)
        
        # Filter for good feedback only
        good_interactions = [
            i for i in interactions 
            if i.get('feedback') == 'good'
        ]
        
        return good_interactions
    
    def should_finetune(self):
        """Check if we have enough new data to finetune"""
        current_count = self.get_current_count()
        new_interactions = current_count - self.last_finetune_count
        
        print(f"[LEARNER] New interactions: {new_interactions}/{self.threshold}")
        
        return new_interactions >= self.threshold
    
    def trigger_finetune(self):
        """Trigger automatic fine-tuning"""
        print(f"\n{'='*60}")
        print(f"🎓 AUTO-FINETUNING TRIGGERED")
        print(f"{'='*60}")
        print(f"Time: {datetime.now()}")
        print(f"New interactions: {self.get_current_count() - self.last_finetune_count}")
        
        # Run fine-tuning
        import subprocess
        result = subprocess.run([
            'python', 'finetune_llm.py',
            '--model', 'Qwen/Qwen2.5-3B-Instruct',
            '--output', f'finetuned_qwen_{datetime.now().strftime("%Y%m%d_%H%M")}'
        ])
        
        if result.returncode == 0:
            print("✅ Auto-finetuning completed successfully!")
            self.last_finetune_count = self.get_current_count()
            self.save_checkpoint()
        else:
            print("❌ Auto-finetuning failed!")
    
    def save_checkpoint(self):
        """Save learning checkpoint"""
        checkpoint = {
            'last_finetune_count': self.last_finetune_count,
            'last_finetune_time': datetime.now().isoformat(),
            'total_finetunings': self.get_total_finetunings() + 1
        }
        
        with open(MODELS_DIR / 'learning_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self):
        """Load learning checkpoint"""
        checkpoint_file = MODELS_DIR / 'learning_checkpoint.json'
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.last_finetune_count = checkpoint.get('last_finetune_count', 0)
    
    def get_total_finetunings(self):
        """Get total number of auto-finetunings"""
        checkpoint_file = MODELS_DIR / 'learning_checkpoint.json'
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f).get('total_finetunings', 0)
        return 0
    
    def monitor_loop(self):
        """Background monitoring loop"""
        print(f"🔄 Continuous learning started (checking every {self.check_interval}s)")
        
        while self.running:
            time.sleep(self.check_interval)
            
            if self.should_finetune():
                self.trigger_finetune()
    
    def start(self):
        """Start continuous learning in background"""
        self.running = True
        self.load_checkpoint()
        
        thread = threading.Thread(target=self.monitor_loop, daemon=True)
        thread.start()
        
        print("✅ Continuous learning enabled!")
        return thread
    
    def stop(self):
        """Stop continuous learning"""
        self.running = False
        print("⏸️ Continuous learning stopped")

# Reinforcement Learning from Human Feedback (RLHF-like)
class RLHFLearner:
    """
    Simplified RLHF: Uses user feedback to weight training examples
    """
    
    def __init__(self):
        self.feedback_weights = {
            'good': 1.0,
            'bad': 0.0,
            'skip': 0.5
        }
    
    def prepare_weighted_dataset(self):
        """Prepare dataset with feedback weights"""
        feedback_file = Path('user_interactions.json')
        if not feedback_file.exists():
            return []
        
        with open(feedback_file, 'r') as f:
            interactions = json.load(f)
        
        weighted_data = []
        for interaction in interactions:
            feedback = interaction.get('feedback', 'skip')
            weight = self.feedback_weights.get(feedback, 0.5)
            
            # Only include if weight > 0
            if weight > 0:
                weighted_data.append({
                    'query': interaction['query'],
                    'response': interaction['response'],
                    'weight': weight,
                    'profile': interaction.get('profile', {})
                })
        
        return weighted_data
    
    def finetune_with_rlhf(self):
        """Fine-tune using weighted examples (RLHF-like)"""
        print("\n🎯 RLHF-style Fine-tuning")
        print("Using user feedback to weight training examples...")
        
        weighted_data = self.prepare_weighted_dataset()
        
        if len(weighted_data) < 10:
            print(f"❌ Not enough feedback data: {len(weighted_data)}/10")
            return
        
        # Save weighted dataset
        with open(MODELS_DIR / 'rlhf_dataset.json', 'w') as f:
            json.dump(weighted_data, f, indent=2)
        
        print(f"✅ Prepared {len(weighted_data)} weighted examples")
        print("📊 Feedback distribution:")
        
        good = sum(1 for d in weighted_data if d['weight'] == 1.0)
        neutral = sum(1 for d in weighted_data if d['weight'] == 0.5)
        
        print(f"   Good: {good}")
        print(f"   Neutral: {neutral}")
        
        # Trigger fine-tuning with weighted data
        import subprocess
        subprocess.run([
            'python', 'finetune_llm.py',
            '--model', 'Qwen/Qwen2.5-3B-Instruct',
            '--output', 'finetuned_rlhf'
        ])

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['continuous', 'rlhf'], default='continuous')
    parser.add_argument('--threshold', type=int, default=50,
                       help='New interactions before auto-finetune')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Check interval in seconds')
    
    args = parser.parse_args()
    
    if args.mode == 'continuous':
        learner = ContinuousLearner(
            threshold=args.threshold,
            check_interval=args.interval
        )
        learner.start()
        
        print("\n📚 Continuous learning is now active!")
        print(f"   Will auto-finetune after {args.threshold} new interactions")
        print(f"   Checking every {args.interval} seconds")
        print("\nPress Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            learner.stop()
            print("\n👋 Stopped")
    
    elif args.mode == 'rlhf':
        rlhf = RLHFLearner()
        rlhf.finetune_with_rlhf()
