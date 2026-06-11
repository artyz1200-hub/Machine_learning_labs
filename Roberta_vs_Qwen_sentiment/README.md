# Transformer Architecture Showdown: Encoder Fine-Tuning vs. Decoder Zero-Shot Prompting

This repository hosts a practical comparative analysis evaluating two fundamentally different Transformer paradigms on a binary sentiment classification task using the **IMDB Movie Reviews dataset**. 

The goal of this project is to move beyond abstract theory and benchmark how an **Encoder-only model (RoBERTa-base)** running a specialized classification head compares against a modern **Decoder-only Language Model (Qwen 2.5 - 1.5B)** working entirely in a zero-shot parameter-frozen context.

---

## 📊 Technical Overview & Setup

* **Hardware Environment:** NVIDIA Tesla T4 GPU (16GB VRAM) via PyTorch CUDA execution.
* **Dataset Profile:** A balanced sampling of the standard Stanford IMDB text benchmark.
    * **Training Samples:** 5,000 text sequences
    * **Test Samples:** 1,000 text sequences (held constant across both evaluation pipelines)
    * **Class Distribution:** Highly balanced (`Negative: 2,477` rows, `Positive: 2,523` rows)

---

## 🚀 Phase 1: The Encoder Pipeline (RoBERTa-base Fine-Tuning)

**RoBERTa (Robustly Optimized BERT Approach)** is designed for deep bidirectional context processing, making it natively exceptional for classification and structural feature extraction. 

### Implementation Strategy
For maximum computational efficiency and clean embedding extraction, the pre-trained `roberta-base` backbone weights were frozen (`requires_grad = False`). A randomly initialized dense linear classification head (`num_labels=2`) was appended to the sequence outputs, trained over a single epoch with dynamic padding.

### Training Progress Logs
```python
Initiating RoBERTa fine-tuning...
# Quantitative Evaluation
# After convergence, the linear head achieved a baseline accuracy of 80.40%.
```
```python
RoBERTa Classification Report:
              precision    recall  f1-score   support

    Negative       0.79      0.85      0.82       515
    Positive       0.83      0.75      0.79       485

    accuracy                           0.80      1000
   macro avg       0.81      0.80      0.80      1000
weighted avg       0.81      0.80      0.80      1000
```

💡 Analysis Note: RoBERTa exhibited a strong capability to isolate negative reviews (Recall: 0.85), though its precision for positive predictions was slightly sharper (0.83). Because the core backbone weights remained frozen, the model relied heavily on its native pre-trained representations, explaining why accuracy plateaued at ~80.4%.

## 🧠 Phase 2: The Decoder Pipeline (Qwen 2.5 Zero-Shot Prompting)
Switching paradigms completely, we deployed Qwen/Qwen2.5-1.5B-Instruct—a 1.5 Billion parameter autoregressive decoder model. This pipeline skips weight updates entirely, optimizing inference latency via strict prompt construction and batched sequence generation.

Implementation Specifics
Padding Configuration: Left-padding explicitly initialized (padding_side = "left"). This prevents positional token drift during batched generation, ensuring newly sampled answer tokens immediately follow the chat template structure.

- Prompt Constraint: * System: "You are an expert NLP sentiment analyzer."

- User: "Analyze the sentiment of this review. Respond with exactly one word: either 'positive' or 'negative'."

- Generation Parameters: max_new_tokens=3, do_sample=False (Greedy decoding for determinism).