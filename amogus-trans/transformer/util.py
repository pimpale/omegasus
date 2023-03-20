# NOTE: Code for training and generation inpired by "Lyrics Generation with GPT-2" by François St-Amant (https://github.com/francoisstamant/lyrics-generation-with-GPT2)

import pandas as pd
import numpy as np
import sklearn
import os
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch.nn.functional import softmax

def get_csv_data(data_dir):
    output_lst = []
    for file in [os.path.join(data_dir, x) for x in os.listdir(data_dir)]:
        with open(file, 'r') as f:
            lines = f.readlines()
            out_str = '|'.join(lines)
            out_str = out_str.replace('\n', '').replace('"', '').replace('|', '\n')
            output_lst.append(out_str)
    return output_lst

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(
    dataset, model,
    batch_size=16, epochs=20, lr=1e-5,
    warmup_steps=200, output_dir=".", output_prefix="gpt",
    save_model_on_epoch=False, save_best_model=False
):

    acc_steps = 100
    device=torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None
    
    train_iterator = trange(
        0, epochs, desc="Epoch"
    )
    
    best_loss = None
    
    loss_history = []

    for epoch in train_iterator:
        print('Loss:', loss)
        epoch_iterator = tqdm(enumerate(train_dataloader), desc="Iteration")
        for idx, entry in epoch_iterator:
            (input_tensor, carry_on, remainder) = pack_tensor(entry[0], input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        
        loss_history.append(loss.item())
        
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
        if save_best_model:
            if best_loss is None or loss < best_loss:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-best.pt"),
                )
                best_loss = loss
    return model, loss_history


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=1,
    entry_length=30, #maximum number of words
    top_p=0.8,
    temperature=0.7,
):
    model.to('cpu')
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
            
            for i in range(entry_length):

                # for i in range(entry_length):
                outputs = model(generated, labels=generated)
                
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                
                if next_token in tokenizer.encode("\t") or next_token in tokenizer.encode("\n") or next_token in tokenizer.encode("<|endoftext|>") or next_token in tokenizer.encode("<|endofsentence|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
        
            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              generated_list.append(output_text)
                
    return generated_list

# def generate_conversation(model, tokenizer, starting_speaker, imposter, who_is_dead, num_rounds = 3):
#     remaining_players = ['Red', 'Green', 'Yellow', 'Blue']
#     remaining_players.remove(who_is_dead)
#     remaining_players.remove(starting_speaker)
#     remaining_players = [starting_speaker] + remaining_players
#     current_convo = ''
#     for _ in range(num_rounds):
#         for player in remaining_players:
#             current_convo = current_convo + player + '\t' + ('imposter' if player == imposter else 'crewmate') + '\t'
#             current_convo = generate(model, tokenizer, current_convo, entry_length=200)[0]
#     return current_convo

def generate_conversation(model, tokenizer, starting_speaker, imposter, who_is_dead, num_rounds = 3):
    remaining_players = ['Red', 'Green', 'Yellow', 'Blue']
    remaining_players.remove(who_is_dead)
    remaining_players.remove(starting_speaker)
    remaining_players = [starting_speaker] + remaining_players
    current_convo = ''
    for _ in range(num_rounds):
        for player in remaining_players:
            player_tag = ('imposter' if player == imposter else 'crewmate')
            response = (player + ' says "' + generate(model, tokenizer, 'A dead body was just found in Communications. All the players have gathered for a meeting. ' + player + ' says "', entry_length=200, temperature=0.7)[0]).split('"')[2]
            current_convo = current_convo + f'{player}\t{player_tag}\t{response}\n'
            
    return current_convo