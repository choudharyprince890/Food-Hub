from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
# loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
# loading the model
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)



# skipping all the special tokens from the output
def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]
    
    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)

        for k, v in tokens_map.items():
            text = text.replace(k, v)

        new_texts.append(text)

    return new_texts

prefix = "items: "

# Parameters that control the length of the output
generation_kwargs = {
    "max_length": 512, # The maximum length the generated tokens can have
    "min_length": 64,   
    "no_repeat_ngram_size": 3,
    "do_sample": True,  # Whether or not to use sampling
    "top_k": 60, # model will only consider the top 60 most probable tokens when generating text. default value for top_k is 50 
    "top_p": 0.95  # it will consider all tokens whose cumulative probability is greater than or equal to 10.95.default value for top_p is 1.0
}

special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

print("these are ---",special_tokens)

# this function will generate the recipe 
def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    # tokenize the input ingredients 
    inputs = tokenizer(inputs,max_length=256,padding="max_length",truncation=True,return_tensors="jax")

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # use model to generate recipe from tokenized output
    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,**generation_kwargs)
    generated = output_ids.sequences
    generated_recipe = target_postprocessing(tokenizer.batch_decode(generated, skip_special_tokens=False),special_tokens)
    return generated_recipe





items = [
    "banana, butter, wine, suger, ice, bread"
]
generated = generation_function(items)
split_data = generated[0].split('\n')
split_data = [item.strip() for item in split_data if item.strip()]

# Create three different lists for title, ingredients, and directions
titles = []
ingredients = []
directions = []

# Loop through the split_data list and populate the three lists accordingly
for item in split_data:
    if item.startswith('title:'):
        titles.append(item.replace('title:', '').strip())
    elif item.startswith('ingredients:'):
        ingredients.append(item.replace('ingredients:', '').strip())
    elif item.startswith('directions:'):
        directions.append(item.replace('directions:', '').strip())

# Print the result
print("Titles:", titles)
print("Ingredients:", ingredients)
print("Directions:", directions)