# 🇳🇵 English to Nepali Neural Machine Translator

A Sequence-to-Sequence Transformer model built from scratch using PyTorch, trained to translate English sentences into Nepali.

## 🚀 Project Overview
This project implements the "Attention Is All You Need" architecture (Vaswani et al., 2017) without relying on pre-built Transformer libraries. The goal was to understand the mathematical mechanics of Self-Attention, Multi-Head Attention, and Positional Encoding by writing them line-by-line.

- **Architecture:** Encoder-Decoder Transformer
- **Framework:** PyTorch
- **Dataset:** English-Nepali Parallel Corpus
- **Training Hardware:** MacBook Air (M3 Chip) via MPS acceleration

## 🧠 Key Features
- **Custom Implementation:** `MultiHeadAttention`, `PositionwiseFeedforward`, and `Encoder/Decoder` layers written from scratch.
- **Data Processing:** Manual vocabulary construction and tokenization. The model parses raw text files to build a custom string-to-index mapping without relying on heavy external tokenizers like spaCy or BERT.
- **Optimization:** Trained for 50 epochs with `Adam` optimizer and `ReduceLROnPlateau`.

## 📊 Results
- **Training Loss:** ~2.45
- **Validation Loss:** ~3.53
- **Sample Output:**
    - *Input:* "Hello how are you?"
    - *Output:* "नमस्कार, तपाईं कसरी हुनुहुन्छ?" (Correct Honorifics!)

## 📂 Repository Structure
- `encoder_decoder_transformer.ipynb`: The main training notebook.
- `experiments/`: Early prototypes and encoder-only experiments.

## 🛠️ How to Run
To run the model, you will need the trained weights (available upon request or via Colab).

```python
# Load model
model.load_state_dict(torch.load('best-model.pt'))
# Translate
translate_sentence("I am going home", model, device)