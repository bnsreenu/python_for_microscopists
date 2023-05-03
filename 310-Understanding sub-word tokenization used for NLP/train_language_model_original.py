# https://youtu.be/dBk98xqEtoA
"""
The purpose of this code is to train the GPT-2 language model from scratch with a 
custom vocabulary based on the input text data, in order to generate text 
that is more relevant to the specific context of the training data.

"""
import tokenizers
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os


# Define the path to the directory containing the input text files
input_dir = "training_files/"


# Define the paths for the output token files and the trained model
output_dir = "output/tokenizer_model"
#model_path = "output/model"
model_path = "output/model"

#Let us first create a customized vocabulary for the training data that the 
# GPT-2 language model will be trained on.

# Initialize the tokenizer with a vocab size of 1000
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the input text files
for filename in os.listdir(input_dir):
    with open(os.path.join(input_dir, filename), "r") as f:
        text = f.read()
        tokenizer.train_from_iterator([text], vocab_size=5000)

# Save the trained customized tokenizer to disk
tokenizer.save_model(output_dir)

"""
GPT2LMHeadModel from the transformers library can only work with tokenizers 
that are of the transformers format. Therefore, the saved tokenizer has to be 
loaded in the transformers format, which is what GPT2TokenizerFast provides.

"""

# Initialize the GPT-2 tokenizer with the same vocab as the ByteLevelBPETokenizer
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(output_dir)

# Set the padding token to [PAD] and add it to the tokenizer's special tokens
gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token



# Initialize the GPT-2 language model with the same vocab as the ByteLevelBPETokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=gpt2_tokenizer.eos_token_id, num_labels=1)
#model.to("cuda")  # Move the model to the GPU

# Define the path to the tokenized input file
input_file = os.path.join(output_dir, "input_data.txt")

# Tokenize the input text files using the GPT-2 tokenizer
with open(input_file, "w") as f:
    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename), "r") as g:
            text = g.read()
            tokens = gpt2_tokenizer.encode(text, add_special_tokens=False)
            for token in tokens:
                f.write(str(token) + " ")
            f.write("\n")

# Load the tokenized input data into a LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=gpt2_tokenizer,
    file_path=input_file,
    block_size=128,
)

# Define the training arguments for the Trainer
training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=10000,
    save_total_limit=2,
    logging_steps=1000,
    logging_dir=model_path,
    
)



# Define the data collator for the Trainer
#The data collator helps to group and batch the individual sequences together 
# into batches of a specified size, and pads the sequences to a uniform length 
# within each batch. The data collator also applies any additional processing 
# steps or modifications to the input data that are necessary for the specific 
# training task or model architecture.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=gpt2_tokenizer, mlm=False,
)

# Initialize the Trainer and train the model on the input data
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# Save the trained language model to disk
model.save_pretrained(model_path)

