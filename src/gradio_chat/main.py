from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Force model to load in 4-bit precision on GPU
model_name = "melnnnnn/just-done-v5"
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(message, history):
    # Tokenize input
    inputs = tokenizer(message, return_tensors="pt").to(device)  # Ensure inputs are on GPU

    # Generate output
    output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)

    # Decode generated text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response

gr.ChatInterface(
    fn=generate_response,
    type="messages"
).launch()
