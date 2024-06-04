import torch
import json
import os
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from perplexity_reward import PerplexityReward
from tqdm import tqdm
import argparse

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def setup_environment(config):
    seed_everything(config['seed'])
    os.environ["WANDB_SHOW_INFO"] = "False"

def initialize_model_and_tokenizer(config, device):
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config['model_name']).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

def prepare_data(config):
    dataset = load_dataset("text", data_files=config['data_files'], split="train").shuffle(seed=config['seed'])
    dataset = dataset.filter(lambda x: len(x["text"]) > 25, batched=False, num_proc=24)
    dataset = dataset.select(range(config['batch_size'] * config['steps']))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataloader

def parse_arguments():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)
    setup_environment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = initialize_model_and_tokenizer(config, device)
    dataloader = prepare_data(config)

    ppo_config = PPOConfig(
        model_name=config['model_name'],
        learning_rate=config['learning_rate'],
        log_with=config['log_with'],
        batch_size=config['batch_size'],
        mini_batch_size=config['mini_batch_size']
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer
    )
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    ppl_reward = PerplexityReward(config['reward_model'])

    for epoch in range(config['epochs']):
        ppl_reward.set_threshold(config['threshold'])
        loop = tqdm(dataloader, leave=True)
        total_steps_done = 0

        for step, batch in enumerate(loop):
            current_mode_step = total_steps_done % config['total_ratio']
            is_clm_step = current_mode_step < config['clm_step_ratio']

            if is_clm_step:
                # CLM STEP
                optimizer.zero_grad()
                input_ids = tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=config["max_length"]
                ).input_ids.to(device)
                
                labels = input_ids.clone()
                outputs = model(input_ids, labels=labels, return_dict=True)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                model.zero_grad()
                
            else: 
                # PPO STEP
                
                # generate queries
                input_ids = tokenizer(batch["text"], return_tensors="pt").input_ids
                input_tensors = [torch.tensor(input_id)[:config['prompt_size']].to(device) for input_id in input_ids]
                queries = tokenizer.batch_decode(input_tensors, skip_special_tokens=True)
                
                # generate responses
                responses = ppo_trainer.generate(input_tensors, **config['generation_kwargs'])
                response_tensors = [response.squeeze()[-config['gen_size']:] for response in responses]
                response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                
                # calculate rewards
                text = [q + r for q, r in zip(queries, response_texts)]
                rewards = ppl_reward(text)
                
                ppo_trainer.step(input_tensors, response_tensors, rewards)

            total_steps_done += 1
            loop.set_description(f"Epoch {epoch + 1}/{config['epochs']} - Step {step + 1}/{config['steps']} - Mode {'CLM' if is_clm_step else 'PPO'}")

        model.save_pretrained(f"{config['output_dir']}/model_{epoch}")
        tokenizer.save_pretrained(f"{config['output_dir']}/model_{epoch}")

if __name__ == "__main__":
    main()
    
