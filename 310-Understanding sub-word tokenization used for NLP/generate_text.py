# https://youtu.be/dBk98xqEtoA
"""

Generate text using our pre-trained model

"""
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model_path = "output/model"
# Load the trained GPT-2 language model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer_model_path = "output/tokenizer_model"
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_model_path)

# Set the model to evaluation mode
model.eval()

# Generate text using the model and tokenizer
input_text = "Life's ultimate question is "
#input_ids = tokenizer.encode(input_text, return_tensors='pt')
input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=True)

output = model.generate(input_ids, max_length=30, do_sample=True)

# Decode the output tokens back to text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

#############



