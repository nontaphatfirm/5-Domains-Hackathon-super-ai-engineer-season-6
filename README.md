# Super AI Engineer Season 6 - 5 Domains Mini Hackathon

This repository contains the Jupyter Notebooks and solutions for the **Super AI Engineer Season 6 (SS6) Hackathon**. The competition was an intensive 24-hour Kaggle sprint covering 5 different AI domains. 

## 🏆 Results Summary

Overall, I successfully passed the baseline for 3 out of the 5 challenges. 

| Domain | Challenge | Score | Status |
| :--- | :--- | :--- | :--- |
| **1. Image Processing** | [House Recognition](https://www.kaggle.com/competitions/super-ai-engineer-season-6-individual-hackathon-house-recognition) | 0.98909 | ✅ Passed Baseline |
| **2. NLP** | [Word Segmentation](https://www.kaggle.com/competitions/super-ai-engineer-ss-6-word-segmentation) | 0.95782 | ❌ Failed Baseline |
| **3. Signal Processing** | [Sleep Staging Classification](https://www.kaggle.com/competitions/super-ai-engineer-ss-6-sleep-stage-classification) | 0.46173 | ✅ Passed Baseline |
| **4. Data Science** | [Heart Disease Prediction](https://www.kaggle.com/competitions/super-ai-engineer-ss-6-heart-disease-prediction) | 0.54716 | ✅ Passed Baseline |
| **5. Vision-Language** | [Thai Language Image Captioning](https://www.kaggle.com/competitions/super-ai-engineer-ss-6-thai-language-image-captioning) | 30.92501 | ❌ Failed Baseline |

---

## 🛠️ Technical Approach & Frameworks

### Domain 1: House Recognition (Image Classification)
* **Frameworks:** FastAI, PyTorch, Pandas
* **Model:** ConvNeXt (`convnext_small_in22k`) via FastAI's `vision_learner`
* **Key Techniques:**
  * **Transfer Learning & Fine-tuning:** Leveraged ImageNet pre-trained weights.
  * **Data Augmentation:** Applied scaling and transformations (`min_scale=0.75`) to prevent overfitting.
  * **DataBlock API:** Standardized image sizing (Resize 460) and ImageNet normalization.

### Domain 2: Thai Word Segmentation (NLP)
* **Frameworks:** PyTorch, PyThaiNLP, Hugging Face Transformers
* **Models:** BiLSTM + CRF, alongside SOTA pre-trained models (DeepCut, AttaCut, Newmm)
* **Key Techniques:**
  * **Sequence Tagging (BIEO):** Formatted data into Begin-Inside-End-Outside token structures.
  * **Safe Chunking:** Split long texts (up to 40k characters) into 200-character chunks at whitespaces to mitigate LSTM vanishing memory issues.
  * **Super Ensemble:** Used Weighted Majority Voting across contextual models (DeepCut) and vocabulary-heavy models (BiLSTM, Newmm).
  * **Custom Dictionary:** Integrated the LST20 training vocabulary directly into Newmm.

### Domain 3: Sleep Stage Classification (Time Series Processing)
* **Frameworks:** PyTorch
* **Model:** Custom LSTM (Long Short-Term Memory) Architecture (`SleepModel`)
* **Key Techniques:**
  * **Sequence Modeling:** Extracted the final hidden state of the LSTM into a Fully Connected Layer.
  * **Class Imbalance Handling:** Implemented `class_weights` within the `CrossEntropyLoss` function to compensate for minority sleep stages.
  * **Optimization:** Used AdamW (Adam with Weight Decay) for better regularization.

### Domain 4: Heart Disease Prediction (Tabular Data)
* **Frameworks:** LightGBM, Scikit-Learn, Pandas
* **Model:** `LGBMClassifier` (Gradient Boosting Decision Trees)
* **Key Techniques:**
  * **Threshold Tuning:** Iteratively searched for the optimal probability threshold (0.1 - 0.6) to maximize the F2 score.
  * **Cross-Validation:** Utilized `StratifiedKFold` (5 Folds) to maintain target distribution and generate Out-Of-Fold (OOF) predictions.
  * **Imbalanced Learning:** Applied `class_weight='balanced'` to force the trees to prioritize positive disease cases.

### Domain 5: Thai Language Image Captioning (Vision-Language)
* **Frameworks:** Hugging Face Transformers, PyTorch, Torchvision
* **Model:** Vision-Encoder-Decoder (Encoder: `ViT-base`, Decoder: `WangchanBERTa`)
* **Key Techniques:**
  * **Cross-Attention Initialization:** Connected the vision and language models to learn multimodal representations.
  * **Data Cleansing:** Filtered out hallucinated sentences and built a custom dictionary to correct common typos in the training set.
  * **Advanced Decoding:** Utilized Beam Search (`num_beams=12`, `length_penalty=1.2`, `no_repeat_ngram_size=3`) to generate fluent Thai captions.
  * **Label Smoothing:** Applied `label_smoothing_factor=0.1` to boost BLEU scores, paired with a Cosine Learning Rate Scheduler.