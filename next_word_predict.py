import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="GRU Next Word Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load resources (cached)
# -----------------------------
@st.cache_resource
def load_resources():
    # Adjust paths if your files are in a different folder
    model_path = r"L:\Softwares\VS Code\Files\RNN\rnn_quote\gru_model.h5"
    tokenizer_path = r"L:\Softwares\VS Code\Files\RNN\rnn_quote\tokenizer.pickle"
    max_length_path = r"L:\Softwares\VS Code\Files\RNN\rnn_quote\max_length.pickle"

    if not all(os.path.exists(p) for p in [model_path, tokenizer_path, max_length_path]):
        st.error("Missing model files! Make sure gru_model.h5, tokenizer.pickle and max_length.pickle are in the same folder.")
        st.stop()

    model = load_model(model_path)
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    with open(max_length_path, "rb") as f:
        max_length = pickle.load(f)
    
    return model, tokenizer, max_length

model, tokenizer, max_length = load_resources()

# -----------------------------
# Prediction function
# -----------------------------
def predict_next_words(text: str, top_k: int = 5):
    """Return top-k next word predictions with probabilities."""
    if not text.strip():
        return []

    # Convert text → sequence
    sequence = tokenizer.texts_to_sequences([text.lower()])
    
    # Pad / truncate to the length the model expects
    padded = pad_sequences(sequence, maxlen=max_length, padding="pre", truncating="pre")
    
    # Predict
    preds = model.predict(padded, verbose=0)[0]
    
    # Get top-k indices
    top_indices = np.argsort(preds)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, "<UNK>")
        prob = float(preds[idx])
        results.append((word, prob))
    
    return results

def generate_text(seed_text: str, num_words: int = 10, temperature: float = 1.0):
    """Generate multiple words sequentially (sampling)."""
    result = seed_text.strip()
    
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([result.lower()])
        padded = pad_sequences(sequence, maxlen=max_length, padding="pre", truncating="pre")
        preds = model.predict(padded, verbose=0)[0]
        
        # Apply temperature
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Sample next word
        next_idx = np.random.choice(len(preds), p=preds)
        next_word = tokenizer.index_word.get(next_idx, "")
        
        if not next_word or next_word == "OOV":
            break
            
        result += " " + next_word
    
    return result

# -----------------------------
# UI
# -----------------------------
st.title("🧠 GRU Next Word Predictor")
st.markdown("Enter some text and get the most likely next words predicted by your GRU model.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of suggestions", min_value=1, max_value=15, value=5)
    temperature = st.slider("Temperature (for generation)", min_value=0.5, max_value=1.5, value=1.0, step=0.1,
                            help="Higher = more creative / random. Lower = more conservative.")
    num_generate = st.slider("Words to generate", min_value=1, max_value=30, value=10)
    
    st.markdown("---")
    st.markdown("**Model Info**")
    st.write(f"- Max sequence length: `{max_length}`")
    st.write(f"- Vocabulary size: `{len(tokenizer.word_index)}`")
    st.write(f"- Embedding dim: `32`")
    st.write(f"- GRU units: `64`")

# Main input
seed_text = st.text_area(
    "Enter your text here:",
    value="the future of artificial intelligence is",
    height=100,
    placeholder="Type a few words..."
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔮 Predict Next Words", use_container_width=True, type="primary")

with col2:
    generate_btn = st.button("✨ Generate Continuation", use_container_width=True)

# -----------------------------
# Results
# -----------------------------
if predict_btn or generate_btn:
    if not seed_text.strip():
        st.warning("Please enter some text first.")
    else:
        if predict_btn:
            with st.spinner("Predicting..."):
                predictions = predict_next_words(seed_text, top_k=top_k)
            
            st.subheader("Top Predictions")
            
            if predictions:
                # Nice progress-bar style display
                for i, (word, prob) in enumerate(predictions, 1):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{i}. {word}**")
                        st.progress(min(prob * 3, 1.0))  # scale for better visibility
                    with col_b:
                        st.metric("Probability", f"{prob*100:.1f}%")
            else:
                st.info("No predictions available.")

        if generate_btn:
            with st.spinner("Generating text..."):
                generated = generate_text(seed_text, num_words=num_generate, temperature=temperature)
            
            st.subheader("Generated Continuation")
            st.success(generated)
            
            # Also show the last predicted word probability context
            st.caption("Tip: Adjust **Temperature** in the sidebar to control creativity.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit + TensorFlow | GRU Next-Word Prediction Model")
