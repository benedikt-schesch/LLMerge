from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
def load_model(model_name):
    """Loads the model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    """Generates a response using the chat template and DeepSeek's system prompt."""
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    MODEL_NAME = "unsloth/deepSeek-r1-distill-qwen-1.5b"
    model, tokenizer = load_model(MODEL_NAME)
    
    prompt = "What is 1+1?"
    response = generate_response(model, tokenizer, prompt)
    
    print("Response:", response)