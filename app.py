from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

class Prompt(BaseModel):
    text: str
    max_length: int = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained("Corianas/1.3b")
model = GPT2LMHeadModel.from_pretrained("Corianas/1.3b").to(device)

@app.post("/generate/")
async def generate_text(prompt: Prompt):
    try:
        inputs = tokenizer.encode(prompt.text, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=prompt.max_length, num_return_sequences=1)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
