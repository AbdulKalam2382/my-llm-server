from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
SAVE_PATH = "./models/flan-t5-base"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(SAVE_PATH)
print("Tokenizer saved!")

print("Downloading model weights...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.save_pretrained(SAVE_PATH)
print("Model saved!")

print("All done! Model is at:", SAVE_PATH)