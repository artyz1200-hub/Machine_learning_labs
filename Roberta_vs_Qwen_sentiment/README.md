
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

```text
RoBERTa Classification Report:
              precision    recall  f1-score   support

    Negative       0.79      0.85      0.82       515
    Positive       0.83      0.75      0.79       485

    accuracy                           0.80      1000
   macro avg       0.81      0.80      0.80      1000
weighted avg       0.81      0.80      0.80      1000

```

> 💡 **Analysis Note:** RoBERTa exhibited a strong capability to isolate negative reviews (Recall: 0.85), though its precision for positive predictions was slightly sharper (0.83). Because the core backbone weights remained frozen, the model relied heavily on its native pre-trained representations, explaining why accuracy plateaued at ~80.4%.

---

## 🧠 Phase 2: The Decoder Pipeline (Qwen 2.5 Zero-Shot Prompting)

Switching paradigms completely, we deployed **Qwen/Qwen2.5-1.5B-Instruct**—a 1.5 Billion parameter autoregressive decoder model. This pipeline skips weight updates entirely, optimizing inference latency via strict prompt construction and batched sequence generation.

### Implementation Specifics

* **Padding Configuration:** Left-padding explicitly initialized (`padding_side = "left"`). This prevents positional token drift during batched generation, ensuring newly sampled answer tokens immediately follow the chat template structure.
* **Prompt Constraint:** * *System:* `"You are an expert NLP sentiment analyzer."`
* *User:* `"Analyze the sentiment of this review. Respond with exactly one word: either 'positive' or 'negative'."`


* **Generation Parameters:** `max_new_tokens=3`, `do_sample=False` (Greedy decoding for determinism).

### Live Peek Into Post-Processing Logic

To evaluate compliance with the structured prompt, raw generated strings were tracked before mapping into numerical labels:

```text
Qwen 2.5 parameters: 1,543,714,304
🚀 Running zero-shot inference on 1000 samples...

🔍 [Live Peek - Sample 1]
   Input text (truncated): 'I wish I had read the comments on IMDb before I saw this movie. The first 1 hour...'
   RAW Output from Qwen:   'negative'
   Parsed Final Label:     Negative (0)

🔍 [Live Peek - Sample 2]
   Input text (truncated): 'I loved this movie! So worth the long running time. I need help with the ending ...'
   RAW Output from Qwen:   'positive'
   Parsed Final Label:     Positive (1)

🔍 [Live Peek - Sample 3]
   Input text (truncated): 'I actually went to see this movie with low expectations since it was the only on...'
   RAW Output from Qwen:   'positive'
   Parsed Final Label:     Positive (1)

📊 Processed 1000/1000
✅ Inference complete in 40.1 seconds.
🎯 Qwen 2.5 Zero-Shot Accuracy: 0.8790

```

### Prompt Compliance & Instruction Following Analysis

One of the primary challenges with decoder deployment is guaranteeing constraint satisfaction. Qwen 2.5 demonstrated exceptional instruction-following adherence, sticking strictly to the requested binary semantic terms **99.9%** of the time.

| Raw Generated Text | Frequency |
| --- | --- |
| `negative` | 597 |
| `positive` | 402 |
| `neutral` | 1 |

> 📌 **Observation:** Only 1 out of 1,000 reviews resulted in an out-of-bounds response (`"neutral"`), which was safely caught by the production post-processing fallback logic and cataloged under the default negative label.

---

## 📈 Visual Benchmarking & Comparative Metrics

When evaluating the models side-by-side on identical text samples, the Decoder (Qwen 2.5) significantly out-performed the frozen-backbone Encoder setup, registering an overall accuracy gap of **+7.50%** (87.90% vs 80.40%).

### Error Matrix Breakdown

* **RoBERTa Head:** Struggled with high False Positives (119 cases where Positive was predicted for a Negative true text). This indicates that the untrained classification head required deeper learning rate adjustments or unfreezing of lower layers to separate weak structural signals.
* **Qwen 2.5 LLM:** Exhibited incredibly clean separations. It only misclassified 19 true negative reviews as positive, highlighting its pre-trained grasp over natural linguistics, sarcasm, and movie review semantics.

---

## 🏁 Architectural Trade-offs & Engineering Conclusions

### Key Takeaways

* **Encoder-only (RoBERTa):** Perfect for high-throughput, low-latency target deployments where dataset sizes are massive enough to justify unfreezing weights and tuning full downstream representations. It remains highly performant with low computational footprints.
* **Decoder-only LLMs (Qwen 2.5):** Highly flexible out-of-the-box. When training examples are restricted or zero-shot rapid prototyping is needed, large pre-trained instruction-tuned LLMs can bypass the training step completely and comfortably dominate performance boundaries out of the gate.
