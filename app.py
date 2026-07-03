import streamlit as st
import os
import datetime
import re
import time
import gc
import torch
from difflib import SequenceMatcher
from llm_model import load_model_and_tokenizer, generate_answer, MODEL_PATH
from hybrid_search import HybridSearchEngine
from backend_exact_match import ExactMatchEngine

# -------------------------------
# OUTPUT DIRECTORY SETUP
# -------------------------------
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 🛡️ CUDA SAFETY LIMITS
# -------------------------------
# Model အများစုအတွက် safe limit များ
MAX_SAFE_INPUT_LENGTH = 1024      # Tokenizer encode လုပ်တဲ့အခါ max length
MAX_SAFE_CONTEXT_LENGTH = 2048    # Context + Question ပေါင်းပြီး max length
MAX_SAFE_NEW_TOKENS = 512         # Generation အတွက် safe limit

def validate_and_truncate_input(tokenizer, question: str, context: str, max_input_length: int = MAX_SAFE_INPUT_LENGTH):
    """
    Input ကို model မတင်ခင် validate လုပ်ပြီး truncate လုပ်ပေးတဲ့ function
    CUDA error မဖြစ်အောင် ကာကွယ်ပေးသည်
    """
    try:
        # Combine question and context
        if context:
            full_text = f"{context}\n\nမေးခွန်း: {question}"
        else:
            full_text = question
        
        # Tokenize and check length
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=False
        )
        
        input_ids = inputs["input_ids"]
        actual_length = input_ids.shape[1]
        
        # 🛡️ Check for out-of-bounds token IDs
        vocab_size = tokenizer.vocab_size
        max_token_id = input_ids.max().item()
        min_token_id = input_ids.min().item()
        
        if max_token_id >= vocab_size or min_token_id < 0:
            #st.warning(f"⚠️ Invalid token IDs detected (min: {min_token_id}, max: {max_token_id}, vocab: {vocab_size}). Truncating problematic tokens...")
            # Clamp token IDs to valid range
            input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
            inputs["input_ids"] = input_ids
        
        # 🛡️ Check sequence length
        if actual_length > MAX_SAFE_CONTEXT_LENGTH:
            st.warning(f"⚠️ Input length ({actual_length}) exceeds safe limit ({MAX_SAFE_CONTEXT_LENGTH}). Truncating...")
            inputs = {k: v[:, :MAX_SAFE_CONTEXT_LENGTH] for k, v in inputs.items()}
            actual_length = MAX_SAFE_CONTEXT_LENGTH
        
        return inputs, actual_length, True  # inputs, length, success
        
    except Exception as e:
        st.error(f"❌ Input validation error: {str(e)}")
        return None, 0, False


def safe_cuda_cleanup():
    """CUDA memory ကို safely ရှင်းလင်းပေးသည်"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    except Exception as e:
        print(f"CUDA cleanup warning: {e}")


def extract_sections(text: str):
    """
    LLM output ထဲက ပုဒ်မတွေကို ဆွဲထုတ်ပေးတဲ့ Function
    """
    if not text:
        return []
    sections = []
    seen = set()
    
    pattern = re.compile(r'(?:ပုဒ်မ|section)\s*([၀-၉0-9]+)(?:\s*\(?([a-zA-Zက-အ])\)?)?(?![၀-၉0-9])', re.IGNORECASE)
    
    for match in pattern.finditer(text):
        num = match.group(1)
        sub = match.group(2)
        
        if sub:
            sec_str = f"ပုဒ်မ {num}"
        else:
            sec_str = f"ပုဒ်မ {num}"
        
        if sec_str not in seen:
            seen.add(sec_str)
            sections.append(sec_str)
            
    return sections

# -------------------------------
# DEDUPLICATION FUNCTION
# -------------------------------
def remove_duplicate_sentences(text: str, threshold: float = 0.8) -> str:
    if not text or not text.strip():
        return text
    parts = text.split('။')
    unique_parts = []
    for part in parts:
        clean_part = part.strip()
        if not clean_part:
            continue
        
        normalized = "  ".join(clean_part.lower().split())
        is_duplicate = False
        
        for existing in unique_parts:
            existing_normalized = "  ".join(existing.lower().split())
            sim = SequenceMatcher(None, normalized, existing_normalized).ratio()
            if sim >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_parts.append(clean_part)

    result = "။  ".join(unique_parts)
    if text.strip().endswith('။') and not result.endswith('။'):
        result += "။ "

    lines = result.split('\n')
    unique_lines = []

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
             continue
        
        normalized_line = "  ".join(clean_line.lower().split())
        is_duplicate_line = False
        
        for existing_line in unique_lines:
            existing_normalized_line = "  ".join(existing_line.lower().split())
            sim = SequenceMatcher(None, normalized_line, existing_normalized_line).ratio()
            if sim >= threshold:
                is_duplicate_line = True
                break
        
        if not is_duplicate_line:
            unique_lines.append(clean_line)

    return "\n ".join(unique_lines)

# -------------------------------
# 🆕 HELPER: RENDER CHAT HISTORY FUNCTION
# -------------------------------
def render_chat_history(messages):
    """
    Chat history ကို render ပေးတဲ့ Function
    Assistant message တွေမှာ section buttons တွေပါ ပါဝင်မယ်
    """
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and not msg.get("is_section_detail", False):
                extracted_sections = msg.get("extracted_sections", [])
                if extracted_sections:
                    st.markdown("**📌 အောက်ပါပုဒ်မတွေကို နှိပ်ပြီး အသေးစိတ်ကြည့်ရှုနိုင်ပါသည်:**")
                    
                    cols = st.columns(min(3, len(extracted_sections)))
                    for i, section in enumerate(extracted_sections):
                        with cols[i % 3]:
                            if st.button(f"{section}", key=f"section_{idx}_{i}", use_container_width=True):
                                st.session_state.clicked_section = section
                                st.rerun()

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="MyanmarGPT-Chat QA Inference",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
.main { background-color: #e6e6e6; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.chat-title { text-align: center; font-size: 32px; font-weight: bold; color: #1c1c1c; margin-bottom: 5px;margin-top:10px }
.chat-subtitle { text-align: center; color: #BBBBBB; margin-bottom: 30px; }
.stChatMessage { border-radius: 15px; padding: 10px; }
div[data-testid="stSidebar"] { background-color: #ccc8c8; }

.section-link-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    cursor: pointer !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-align: left !important;
    display: inline-flex !important;
    align-items: center !important;
    margin: 4px 8px 4px 0 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
}
.section-link-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
}
.section-link-btn:active {
    transform: translateY(0) !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# MODEL & ENGINE LOADING (CACHED)
# -------------------------------
@st.cache_resource
def load_llm_cached(model_path: str):
    with st.spinner("Loading LLM Model and Tokenizer... Please wait."):
        return load_model_and_tokenizer(model_path)

@st.cache_resource
def load_search_engine_cached():
    with st.spinner("Loading Hybrid Search Engine (LABSE + FAISS)... Please wait."):
        return HybridSearchEngine(
            exact_json_path="dataset_definition.json",
            #labse_model_path=r"C:\Users\MSI\OneDrive\Desktop\Implement\MyanmarGPT\Latest_myanmargpt\Forthird_seminar\latest_retrieval\labse-finetuned",
            labse_model_path=r"./labse-finetuned0207",
            faiss_path=r".\faiss_index၁",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

@st.cache_resource
def load_exact_match_engine_cached():
    with st.spinner("Loading Exact Match Engine... Please wait."):
        return ExactMatchEngine(json_path="dataset_definition.json")

tokenizer, model = load_llm_cached(MODEL_PATH)

# 🛡️ Get model's actual max length for safety checks
MODEL_MAX_LENGTH = getattr(model.config, 'max_position_embeddings', 2048)
MODEL_VOCAB_SIZE = getattr(model.config, 'vocab_size', tokenizer.vocab_size)
st.session_state['model_max_length'] = MODEL_MAX_LENGTH

search_engine = load_search_engine_cached()
exact_match_engine = load_exact_match_engine_cached()

# -------------------------------
# SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_section_query" not in st.session_state:
    st.session_state.pending_section_query = None
if "clicked_section" not in st.session_state:
    st.session_state.clicked_section = None

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    if st.button("🔄 New Chat"):
        st.session_state.messages = []
        st.session_state.pending_section_query = None
        st.session_state.clicked_section = None
        safe_cuda_cleanup()
        st.success("✨ Cleared successfully!")
        st.rerun()

    st.divider()

    st.subheader("🔍 RAG Parameters")
    use_rag = st.checkbox("Use Automatic RAG Retrieval", value=True)
    top_k = st.slider("RAG Top-K Retrieval", min_value=1, max_value=10, value=3)
    
    # 🛡️ Safe limits for context tokens
    max_context_tokens = st.slider(
        "Max Context Tokens (LLM Limit)",
        min_value=256,
        max_value=min(2048, MODEL_MAX_LENGTH),  # 🛡️ Model's actual limit
        value=512,
        step=128,
        help="If retrieved context exceeds this, it will be safely truncated to prevent LLM errors."
    )

    st.divider()

    st.subheader("🤖 LLM Generation Parameters")
    
    # 🛡️ Safe limits for generation
    max_new_tokens = st.slider(
        "Max New Tokens", 
        min_value=50, 
        max_value=min(MAX_SAFE_NEW_TOKENS, 1024),  # 🛡️ Capped at safe limit
        value=300, 
        step=10
    )
    
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
    top_p = st.slider("Top-p (Nucleus Sampling)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    do_sample = st.checkbox("Do Sample", value=True)
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

    # 🛡️ Show model info
    st.divider()
    st.subheader("ℹ️ Model Info")
    st.info(f"""
    **Max Position Embeddings:** {MODEL_MAX_LENGTH}
    **Vocab Size:** {MODEL_VOCAB_SIZE}
    **Safe Input Limit:** {MAX_SAFE_INPUT_LENGTH}
    **Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}
    """)

    st.divider()

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    '<div class="chat-title">(၂၀၀၈)ဖွဲ့စည်းပုံအခြေခံဥပဒေဆိုင်ရာ အမေး/အဖြေများ မေးမြန်းနိုင်ပါသည်။</div>',
    unsafe_allow_html=True
)

# -------------------------------
# 🆕 HANDLE CLICKED SECTION
# -------------------------------
if st.session_state.clicked_section:
    section_query = st.session_state.clicked_section
    st.session_state.clicked_section = None
    
    st.session_state.messages.append({
        "role": "user",
        "content": section_query
    })

    with st.spinner(f"🔍 {section_query} ကို ရှာဖွေနေသည်..."):
        res = exact_match_engine.search(query=section_query)
        
        if "results" in res and res["results"]:
            formatted_result = ""
            for r in res["results"]:
                formatted_result += f"**{r.get('section', section_query)}**\n\n{r.get('context', '')}\n\n"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_result.strip(),
                "is_section_detail": True,
                "extracted_sections": []
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {section_query} အချက်အလက်ကို Database တွင် ရှာမတွေ့ပါ။",
                "is_section_detail": True,
                "extracted_sections": []
            })

    st.rerun()

# -------------------------------
# 🆕 HANDLE PENDING SECTION QUERY
# -------------------------------
if st.session_state.pending_section_query:
    section_query = st.session_state.pending_section_query
    st.session_state.pending_section_query = None
    
    st.session_state.messages.append({
        "role": "user",
        "content": section_query
    })

    with st.spinner(f"🔍 {section_query} ကို ရှာဖွေနေသည်..."):
        res = exact_match_engine.search(query=section_query)
        
        if "results" in res and res["results"]:
            formatted_result = ""
            for r in res["results"]:
                formatted_result += f"**{r.get('section', section_query)}**\n\n{r.get('context', '')}\n\n"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_result.strip(),
                "is_section_detail": True,
                "extracted_sections": []
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {section_query} အချက်အလက်ကို Database တွင် ရှာမတွေ့ပါ။",
                "is_section_detail": True,
                "extracted_sections": []
            })

    st.rerun()

# -------------------------------
# RENDER HISTORY
# -------------------------------
render_chat_history(st.session_state.messages)

# -------------------------------
# CHAT INPUT & GENERATION LOGIC
# -------------------------------
user_input = st.chat_input("မေးခွန်းကို ဒီမှာရေးပါ...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        final_text = ""
        extracted_sections = []
        
        try:
            start_time = time.time()
            
            # 🛡️ Pre-generation CUDA cleanup
            safe_cuda_cleanup()
            
            with st.spinner("Processing..."):
                question = user_input
                context = ""
                
                section_pattern = re.compile(r'(?:ပုဒ်မ|အခန်း|section|chapter|ပုဒ်)', re.IGNORECASE)
                is_section_query = bool(section_pattern.search(question))
                
                analysis_keywords = [
                    'ကွာခြားချက်', 'ကွာခြား', 'နှိုင်းယှဉ်ချက်', 'နှိုင်းယှဉ်',
                    'သုံးသပ်ချက်', 'သုံးသပ်', 'တူညီချက်', 'တူညီ',
                    'ဆက်စပ်ချက်', 'ဆက်စပ်', 'ဘာကြောင့်', 'အဘယ်ကြောင့်',
                    'သက်ရောက်မှု', 'အကျိုးဆက်', 'အကြောင်းရင်း', 'အဘယ်သို့','ဘယ်နှစ်ခု','ဘယ်နှစ်ခန်း','ဘယ်နှခု','ဘယ်နှခန်း',
                    'ကောင်းကျိုး', 'ဆိုးကျိုး', 'အားသာချက်', 'အားနည်းချက်', 'အကဲဖြတ်','ဝေဖန်',
                    'difference', 'compare', 'comparison', 'analyze', 'analysis',
                    'similar', 'different', 'why', 'because', 'effect'
                ]
                analysis_pattern = re.compile('|'.join(map(re.escape, analysis_keywords)), re.IGNORECASE)
                is_analysis_query = bool(analysis_pattern.search(question))
                
                use_exact_match_only = is_section_query and not is_analysis_query
                
                final_context = ""
                raw_answer = ""
                cleaned_answer = ""
                prompt_text = ""
                skip_llm = False
                
                if use_exact_match_only:
                    search_results = exact_match_engine.search(query=question)
                
                    if "error" in search_results:
                        cleaned_answer = f"⚠️ {search_results['error']}"
                        raw_answer = cleaned_answer
                        prompt_text = "Exact Match Retrieval (No LLM Generation)"
                    else:
                        results = search_results.get("results", [])
                        missing = search_results.get("missing", [])
                        
                        if results:
                            all_raw_texts = [res.get("context", "") for res in results if res.get("context")]
                            final_context = "\n\n".join(all_raw_texts)
                            final_context = remove_duplicate_sentences(final_context, threshold=0.8)
                            
                            # 🛡️ Truncate context if too long
                            if len(final_context) > MAX_SAFE_CONTEXT_LENGTH * 4:  # Rough char estimate
                                final_context = final_context[:MAX_SAFE_CONTEXT_LENGTH * 4]
                            
                            formatted_answer = ""
                            for res in results:
                                formatted_answer += f"**{res.get('section', 'Unknown Section')}**\n{res.get('context', '')}\n\n"
                            
                            if missing:
                                formatted_answer += f"\n⚠️ *Missing sections in database: {', '.join(map(str, missing))}*"
                            
                            cleaned_answer = formatted_answer
                            raw_answer = final_context
                            prompt_text = "Exact Match Retrieval (No LLM Generation)"
                        else:
                            cleaned_answer = "❌ No relevant sections found."
                            raw_answer = cleaned_answer
                            prompt_text = "Exact Match Retrieval (No LLM Generation)"
                else:
                    final_context = context
                    
                    if use_rag:
                        search_results = search_engine.search(query=question, top_k=top_k)
                        
                        if "error" in search_results and not search_results.get("results"):
                            st.warning(f"⚠️ Search Warning: {search_results.get('error', 'No results found')}")
                        
                        if search_results.get("results"):
                            all_raw_texts = []
                            for res in search_results["results"]:
                                raw_text = res.get("context", res.get("content", ""))
                                if raw_text:
                                    all_raw_texts.append(raw_text)
                            
                            combined_raw_context = "\n\n".join(all_raw_texts)
                            final_context = remove_duplicate_sentences(combined_raw_context, threshold=0.7)
                            
                            # 🛡️ Truncate context if too long
                            if len(final_context) > max_context_tokens * 4:  # Rough char to token estimate
                                final_context = final_context[:max_context_tokens * 4]
                        else:
                            skip_llm = True
                            cleaned_answer = "⚠️ သင့်မေးသော မေးခွန်းကို ပြန်လည်ပြင်ဆင်မေးပေးပါရန်။\n\nကျေးဇူးပြု၍ မေးခွန်းကို ပိုမိုတိကျစွာ ပြန်လည်ရေးသားပေးပါ။"
                            raw_answer = cleaned_answer
                            prompt_text = "No Context Found (LLM Skipped)"
                    
                    if not skip_llm:
                        # 🛡️ VALIDATE INPUT BEFORE PASSING TO MODEL
                        with st.spinner("🛡️ Validating input for safe generation..."):
                            validated_inputs, input_length, is_valid = validate_and_truncate_input(
                                tokenizer, question, final_context, 
                                max_input_length=min(max_context_tokens, MAX_SAFE_INPUT_LENGTH)
                            )
                        
                        if not is_valid or validated_inputs is None:
                            skip_llm = True
                            cleaned_answer = "⚠️ Input validation failed. Please try a shorter question or reduce context length."
                            raw_answer = cleaned_answer
                            prompt_text = "Input Validation Failed"
                        else:
                            # 🛡️ Adjust max_new_tokens based on available space
                            available_tokens = MODEL_MAX_LENGTH - input_length
                            safe_max_new_tokens = min(max_new_tokens, max(50, available_tokens - 50))
                            
                            if safe_max_new_tokens < 50:
                                skip_llm = True
                                cleaned_answer = f"⚠️ Input too long ({input_length} tokens). Model has {MODEL_MAX_LENGTH} token limit. Please shorten your question or context."
                                raw_answer = cleaned_answer
                                prompt_text = "Input Too Long"
                            else:
                                # 🛡️ Safe generation with CUDA error handling
                                try:
                                    prompt_text, raw_answer, cleaned_answer = generate_answer(
                                        tokenizer, model, question, final_context,
                                        safe_max_new_tokens, temperature, top_p,
                                        do_sample, repetition_penalty, 
                                        max_context_tokens=min(max_context_tokens, MAX_SAFE_CONTEXT_LENGTH)
                                    )
                                except torch.cuda.CudaError as cuda_err:
                                    # 🛡️ Catch CUDA-specific errors
                                    safe_cuda_cleanup()
                                    cleaned_answer = f"⚠️ CUDA Error occurred. The model encountered a memory or computation issue.\n\n**Error Details:** {str(cuda_err)}\n\n**Suggestions:**\n- Try reducing Max Context Tokens\n- Try reducing Max New Tokens\n- Clear chat and try again"
                                    raw_answer = cleaned_answer
                                    prompt_text = "CUDA Error (Handled)"
                                    st.error(f"CUDA Error: {str(cuda_err)}")
                                except RuntimeError as rt_err:
                                    # 🛡️ Catch general runtime errors (often CUDA related)
                                    safe_cuda_cleanup()
                                    if "cuda" in str(rt_err).lower() or "device" in str(rt_err).lower():
                                        cleaned_answer = f"⚠️ Device Error occurred during generation.\n\n**Error:** {str(rt_err)}\n\n**Suggestions:**\n- Reduce input length\n- Clear chat and restart\n- Try CPU mode if available"
                                        raw_answer = cleaned_answer
                                        prompt_text = "Device Error (Handled)"
                                        st.error(f"Runtime Error: {str(rt_err)}")
                                    else:
                                        raise  # Re-raise if not CUDA related
                            
        except torch.cuda.CudaError as e:
            elapsed = time.time() - start_time
            safe_cuda_cleanup()
            final_text = f"❌ **CUDA Error:** {str(e)}\n\n⏱️ **Response time:** `{elapsed:.2f}` sec\n\n**🛡️ Suggestions:**\n1. Reduce Max Context Tokens in sidebar\n2. Reduce Max New Tokens\n3. Try a shorter question\n4. Click 'New Chat' to clear memory"
            placeholder.markdown(final_text, unsafe_allow_html=True)
            extracted_sections = []
            st.error(f"CUDA Error caught: {str(e)}")
            
        except RuntimeError as e:
            elapsed = time.time() - start_time
            safe_cuda_cleanup()
            if "cuda" in str(e).lower() or "device-side assert" in str(e).lower():
                final_text = f"❌ **Device Error:** {str(e)}\n\n⏱️ **Response time:** `{elapsed:.2f}` sec\n\n**🛡️ This is usually caused by:**\n- Input sequence too long\n- Invalid token IDs\n- GPU memory overflow\n\n**Please try:**\n1. Reducing context/new tokens in sidebar\n2. Asking a shorter question\n3. Clicking 'New Chat'"
            else:
                final_text = f"❌ **Error:** {str(e)}\n\n⏱️ **Response time:** `{elapsed:.2f}` sec"
            placeholder.markdown(final_text, unsafe_allow_html=True)
            extracted_sections = []
            st.error(f"Runtime Error: {str(e)}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            safe_cuda_cleanup()
            final_text = f"❌ **Error:** {str(e)}\n\n⏱️ **Response time:** `{elapsed:.2f}` sec"
            placeholder.markdown(final_text, unsafe_allow_html=True)
            extracted_sections = []
            st.error(f"Unexpected Error: {str(e)}")
        
        else:
            # Only run this if no exception occurred
            if not final_text:  # If final_text wasn't set by error handlers
                try:
                    elapsed = time.time() - start_time
                    
                    # 🆕 EXTRACT SECTIONS FROM LLM OUTPUT
                    extracted_sections = extract_sections(cleaned_answer)
                    
                    # Add response time to answer
                    cleaned_answer += f"\n\n⏱️ **Response time:** `{elapsed:.2f}` sec"
                                
                    # Stream the answer
                    for char in cleaned_answer:
                        final_text += char
                        placeholder.markdown(final_text + "▌", unsafe_allow_html=True)
                        time.sleep(0.002)
                    placeholder.markdown(final_text, unsafe_allow_html=True)
                    
                    # 🆕 RENDER SECTION BUTTONS
                    if extracted_sections:
                        st.markdown("**📌 အောက်ပါပုဒ်မတွေကို နှိပ်ပြီး အသေးစိတ်ကြည့်ရှုနိုင်ပါသည်:**")
                        cols = st.columns(min(3, len(extracted_sections)))
                        for i, section in enumerate(extracted_sections):
                            with cols[i % 3]:
                                if st.button(f"{section}", key=f"section_current_{i}", use_container_width=True):
                                    st.session_state.clicked_section = section
                                    st.rerun()
                    
                    # Save to file
                    if cleaned_answer and cleaned_answer not in ["No results found.", "❌ No relevant sections found."]:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"rag_output_{timestamp}.txt"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        
                        save_content = f"=== MyanmarGPT-Chat QA Output ===\n"
                        save_content += f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        save_content += f"Question:\n{question}\n\n"
                        save_content += f"Intent: Section={is_section_query}, Analysis={is_analysis_query}\n"
                        save_content += f"Routing: {'Exact Match (LLM Bypass)' if use_exact_match_only else ('No Context (LLM Skipped)' if skip_llm else 'RAG + LLM')}\n\n"
                        save_content += f"Retrieved Context (Deduplicated):\n{final_context if (use_rag or use_exact_match_only) else context}\n\n"
                        save_content += f"Generated Answer:\n{cleaned_answer}\n"
                        save_content += f"Response Time: {elapsed:.2f} sec\n"
                        save_content += f"================================="
                        
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(save_content)
                    
                    # 🛡️ Post-generation cleanup
                    safe_cuda_cleanup()
                    
                except Exception as inner_e:
                    elapsed = time.time() - start_time
                    safe_cuda_cleanup()
                    final_text = f"❌ **Post-processing Error:** {str(inner_e)}\n\n⏱️ **Response time:** `{elapsed:.2f}` sec"
                    placeholder.markdown(final_text, unsafe_allow_html=True)
                    extracted_sections = []
        
        # 🆕 Append message with extracted_sections
        if final_text:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_text,
                "extracted_sections": extracted_sections,
                "is_section_detail": False
            })