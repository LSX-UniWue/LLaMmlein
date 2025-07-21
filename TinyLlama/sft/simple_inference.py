from transformers import AutoTokenizer
import transformers 
import torch
model = "/Users/juliawunderle/Desktop/checkpoint-9750"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Wofür steht DNA?"
formatted_prompt = (
    f"### Human: {prompt} ### Assistant:"
)


sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=60,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
