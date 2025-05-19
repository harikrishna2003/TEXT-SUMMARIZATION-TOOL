import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import os
import torch
import re
from transformers import BartTokenizer, BartForConditionalGeneration

def clean_text(text):
    # Remove non-ASCII chars and extra whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra whitespace
    return text

@st.cache_resource(show_spinner=True)
def load_model():
    # Use absolute path for your fine-tuned BART model directory
    save_directory = os.path.abspath(r'c:\Users\rhabp\Documents\internship\task1\task1\bart-xsum-finetuned')

    if not os.path.exists(save_directory):
        st.error(f"Model directory not found: {save_directory}")
        return None, None, None

    tokenizer = BartTokenizer.from_pretrained(save_directory)
    model = BartForConditionalGeneration.from_pretrained(save_directory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

def generate_summary(input_text, max_length=256, tokenizer=None, model=None, device=None):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=int(max_length * 0.75),  # Minimum 75% of max length
        num_beams=8,
        early_stopping=True,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    st.title("BART Summarization App")

    with st.spinner("Loading model, please wait..."):
        tokenizer, model, device = load_model()

    if tokenizer is None or model is None:
        return  # Stop if model is not loaded

    text = st.text_area("Enter text to summarize:", height=200)
    max_len = st.slider("Max summary length", min_value=50, max_value=512, value=256, step=10)

    if st.button("Generate Summary") and text.strip():
        cleaned_text = clean_text(text)
        try:
            summary = generate_summary(cleaned_text, max_length=max_len,
                                       tokenizer=tokenizer, model=model, device=device)
            st.subheader("Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error during summary generation: {e}")

if __name__ == "__main__":
    main()
