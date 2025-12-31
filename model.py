import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-2-2b-it"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_device():
    # Apple Silicon: MPS if available, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
logger.info(f"Using device: {device.type}")
logger.info(f"Loading LLM model: {MODEL_ID}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Some tokenizers/models don't define pad_token; set it safely
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model (FIX: torch_dtype, not dtype)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()

logger.info("Model loaded successfully")


@torch.inference_mode()
def ask_llm(question: str, max_new_tokens: int = 120) -> str:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    # Gemma Instruct uses chat template (user -> assistant)
    messages = [{"role": "user", "content": question}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Only decode the generated continuation
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
