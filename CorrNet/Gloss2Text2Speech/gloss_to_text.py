import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Paths to the files (please customise)
adapter_model_path = "/Users/beni/projects/CorrNet/Gloss2Text2Speech/pretrained"
adapter_config_path = "/Users/beni/projects/CorrNet/Gloss2Text2Speech/pretrained"

print("📢 Loading the Gloss-to-Text model...")

# Load the adapter
base_model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# Load the adapter configuration and the model
model.load_adapter(adapter_model_path)

print("✅ Model and adapter successfully loaded!")

def gloss_to_text(gloss_sentence):
    """
    Converts a list of glosses into a natural language sentence.

    :param gloss_sentence: String containing glosses (e.g., "MONDAY MORE CLOUD RAIN")
    :return: Generated natural sentence
    """

    print("\n📢 **Input Glosses:**", gloss_sentence)

    # **Tokenize & pass to model**
    inputs = tokenizer(gloss_sentence, return_tensors="pt")
    output = model.generate(inputs.input_ids, max_length=50)

    # **Decode & return the result**
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("📝 **Generated Text:**", output_text)

    return output_text
