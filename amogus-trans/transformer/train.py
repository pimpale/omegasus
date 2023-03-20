import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from osus_dataset import OSUSDataset
from util import get_csv_data, train, generate, generate_conversation, generate_conversation_base_model
import torch

DATA_DIR = '../data_dir'
MODEL_DIR = 'model_dir'
SAVED_MODEL_PREFIX = 'bigboi'
    
def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    train_data = get_csv_data(os.path.join(DATA_DIR, 'train'))
    
    train_dataset = OSUSDataset(train_data, tokenizer, 1024) # OSUSDataset Obj
    
    model = train(train_dataset, 
                  model, 
                  batch_size=1,
                  epochs=20, 
                  lr=2e-5,
                  output_dir=MODEL_DIR,
                  output_prefix=SAVED_MODEL_PREFIX,
                  save_model_on_epoch=True,
                  save_best_model=True
                  )
    
    # print(train_data[10])

    # generated_message = generate(model, tokenizer, "Red\tcrewmate\t", entry_count=10, entry_length=100)

if __name__ == '__main__':
    main()