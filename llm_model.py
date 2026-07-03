import torch
import re
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"C:\Users\MSI\OneDrive\Desktop\Implement\MyanmarGPT\Latest_myanmargpt\Forthird_seminar\latest_retrieval\08062026\finetuned-myanmarGPT-chat-qa"
#MODEL_PATH = r".\finetuned-myanmarGPT-chat_2806"
# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def remove_references_and_sections(text: str) -> str:
    if not text: 
        return text
    
    # 1. Remove "Section 123" or "section 123(a)"
    text = re.sub(r'(?i)\bsection\s+\d+[a-zA-Z0-9()\-\.]*', '', text)
    
    # 2. Remove "References: ..." or "References - ..."
    text = re.sub(r'(?i)references?\s*[:\-].*$', '', text, flags=re.DOTALL)
    
    # 3. Remove citation brackets like "[1]", "[1, 2]", "[1. 2]" 
    # ✅ FIXED: Added proper escaping \[ and \] to prevent "unbalanced parenthesis" error
    text = re.sub(r'\[\d+(?:[,.\s]*\d+)*\]', '', text)
    
    # 4. Remove "(see section ...)"
    text = re.sub(r'\(\s*(?i)see\s+section[^)]*\)', '', text)
    
    # 5. Remove multiple newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 6. Remove multiple spaces
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()

def remove_duplicate_sentences(text: str, threshold: float = 0.8) -> str:
    if not text or not text.strip():
        return text
    
    parts = text.split('။')
    unique_parts = []
    
    for part in parts:
        clean_part = part.strip()
        if not clean_part:
            continue
            
        normalized = " ".join(clean_part.lower().split())
        is_duplicate = False
        
        for existing in unique_parts:
            existing_normalized = " ".join(existing.lower().split())
            sim = SequenceMatcher(None, normalized, existing_normalized).ratio()
            if sim >= threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_parts.append(clean_part)
            
    result = "။ ".join(unique_parts)
    if text.strip().endswith('။') and not result.endswith('။'):
        result += "။"
        
    lines = result.split('\n')
    unique_lines = []
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        normalized_line = " ".join(clean_line.lower().split())
        is_duplicate_line = False
        
        for existing_line in unique_lines:
            existing_normalized_line = " ".join(existing_line.lower().split())
            sim = SequenceMatcher(None, normalized_line, existing_normalized_line).ratio()
            if sim >= threshold:
                is_duplicate_line = True
                break
                
        if not is_duplicate_line:
            unique_lines.append(clean_line)
            
    return "\n".join(unique_lines)

def truncate_to_last_myanmar_period(text: str) -> str:
    if not text:
        return text
    last_period_idx = text.rfind("။")
    if last_period_idx != -1:
        return text[:last_period_idx + 1].strip()
    return text.strip()

def truncate_context_by_tokens(tokenizer, context: str, max_tokens: int = 2048) -> str:
    if not context:
        return context
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return context
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    search_start = int(len(truncated_text) * 0.8)
    last_period = truncated_text.rfind("။", search_start)
    if last_period != -1:
        return truncated_text[:last_period + 1]
    return truncated_text

def build_prompt(question: str, context: str) -> str:
    prompt_parts = []
    if question:
        prompt_parts.append(f"Question:\n{question}")
    if context:
        prompt_parts.append(f"Context:\n{context}")
    prompt_text = "\n\n".join(prompt_parts)
    return f"<|user|>\n{prompt_text}\n<|assistant|>\n"

# -------------------------------
# MODEL LOADING
# -------------------------------
def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("⚠️ pad_token is None. Setting pad_token = eos_token.")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("⚠️ pad_token and eos_token are None. Added [PAD] token.")
            
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True
    )

    if tokenizer.pad_token == '[PAD]':
        model.resize_token_embeddings(len(tokenizer))
        
    model.eval() 
    return tokenizer, model

# -------------------------------
# GENERATION LOGIC
# -------------------------------
def generate_answer(tokenizer, model, question: str, context: str,
                    max_new_tokens: int, temperature: float, top_p: float,
                    do_sample: bool, repetition_penalty: float, max_context_tokens: int = 2048):
    if context:
        context = truncate_context_by_tokens(tokenizer, context, max_tokens=max_context_tokens)

    prompt_text = build_prompt(question, context)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_token_id

    if pad_token_id is None:
        pad_token_id = 0 
    if eos_token_id is None:
        eos_token_id = pad_token_id

    max_id = inputs["input_ids"].max().item()
    vocab_size = model.config.vocab_size
    if max_id >= vocab_size:
        raise ValueError(f"🚨 CUDA Error likely caused by Token ID {max_id} exceeding Vocab Size {vocab_size}! Check your tokenizer and model match.")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty 
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    cleaned_answer = remove_references_and_sections(raw_answer)
    cleaned_answer = remove_duplicate_sentences(cleaned_answer, threshold=0.8)
    cleaned_answer = truncate_to_last_myanmar_period(cleaned_answer)

    return prompt_text, raw_answer, cleaned_answer