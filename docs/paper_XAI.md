# LEARNING AUDIO CONCEPTS FROM COUNTERFACTUAL NATURAL LANGUAGE 

**Authors:** Ali Vosoughi*, Luca Bondi, Ho-Hsiang Wu, Chenliang Xu 
**Affiliations:** University of Rochester, Rochester, NY, USA; Bosch Research, USA - Bosch Center for Artificial Intelligence 

---

## ABSTRACT
Conventional audio classification relied on predefined classes, lacking the ability to learn from free-form text.  Recent methods unlock learning joint audio-text embeddings from raw audio-text pairs describing audio in natural language.  Despite recent advancements, there is little exploration of systematic methods to train models for recognizing sound events and sources in alternative scenarios, such as distinguishing fireworks from gunshots at outdoor events in similar situations.  This study introduces causal reasoning and counterfactual analysis in the audio domain.  We use counterfactual instances and include them in our model across different aspects.  Our model considers acoustic characteristics and sound source information from human-annotated reference texts.  To validate the effectiveness of our model, we conducted pre-training utilizing multiple audio captioning datasets.  We then evaluate with several common downstream tasks, demonstrating the merits of the proposed method as one of the first works leveraging counterfactual information in audio domain.  Specifically, the top-1 accuracy in open-ended language-based audio retrieval task increased by more than 43%. 

**Index Terms:** sound event detection, audio understanding, multimodal representations, free-form text, counterfactual representation learning, audio captioning 

---

## 1. INTRODUCTION
Conventional audio processing in machine learning relies on predefined categories, limiting the ability of the models to understand audio nuances using descriptive text.  These constraints limit open-ended and contrastive training for better audio-text alignment.  New trends in the field improve classic models using audio-text learning from audio data and matching natural language descriptions that have become successful models in image-text tasks.  Learning audio representations from pairs of audio and their textual descriptions facilitates the development of foundational models for audio tasks, enabling audio-text models to generalize beyond the confines of predefined classes, leveraging natural language descriptions.  Advancements in audio-text representation include AudioCLIP, a tri-modal model, and Wav2CLIP, which extends CLIP to audio.  Subsequently, Elizalde et al. proposed CLAP (Contrastive Language-Audio Pretraining), a method that takes its inspiration from successful image-text models.  CLAP is able to train on audio and text directly without relying on image data for the learning process.  

However, human annotated audio captioning data is expensive and time-consuming to acquire.  Recently, with large language models (LLMs) such as ChatGPT, an upgrade version of GPT-3 fine-tuned to follow human instructions, have become popular and been utilized to augment learning for various domains, both LAION-Audio-630K and WavCaps leverage the powerful textual re-writing and editing capabilities in order to acquire more audio-text pairs.  Furthermore, methods such as Pengi and listen, think, and understand have built upon these foundational methodologies.  

Distinguishing between sounds in similar conditions, such as the sound of a firework and a gunshot in the same concert event, requires controlled trials of both sounds and a learning algorithm to differentiate between them in the same context.  However, existing audio-text datasets lack alternative scenarios for ethical and practical reasons.  Counterfactual reasoning has been utilized to improve multimodal models involving vision modality and to aid grounding concepts within visual objects.  The semantic differences between pairs of similar but slightly different audio clips have been discussed in prior works, addressing audio captioning.  

To the best of our knowledge, our work is pioneering in utilizing the knowledge base and capabilities of LLMs to integrate counterfactual reasoning.  This provides data and methods to enhance the learning of audio-text correspondence with counterfactual information.  The proposed method utilizes counterfactual language and multimodal embeddings to improve textual discriminability in audio processing tasks.  Subsequently, we develop a composite loss function that integrates the concept of triplet angular spaces and enforces factual audio-text consistency.  

We are inspired by recent developments in causality using LLMs that offer new possibilities, such as augmentation of meaningful counterfactuals in situations where audio learning data may not be available.  The potential of the proposed method is to influence a wide array of applications, including automatic speech recognition, sound event detection, and audio-visual scene understanding.  The core innovation of our work lies in being the first to integrate causal and counterfactual models into the analysis of audio data. 

---

## 2. LEARNING AUDIO FROM COUNTERFACTUAL
Causality describes the relationship between a cause and its subsequent effect.  This study focuses on the causal relationships within audio samples that can be inferred from their corresponding text captions.  Specifically, we address scenarios where data acquisition of counterfactual audio is impossible or costly by leveraging natural language as a substitute for imaginative data.  Our method aligns with recent advancements in causality in natural language that aim to extend domains of causality to LLMs.  We first explore prior work of CLAP that motivated us and then propose our method as the first work to extend counterfactuals to audio learning. 

### Contrastive Language-Audio Pretraining (CLAP)
Given a batch of N pairs $(x_{i},y_{i})$, $x_{i}$ represents an audio sample and $y_{i}$ is its corresponding text caption for $i=1,...,N$.  We define the audio and text encoders as $E_{a}(i)=\phi_{audio}(x_{i})$ and $E_{t}(i)=\phi_{text}(y_{i})$, respectively.  These encoders transform each $x_{i}$ and y into vectors in $\mathbb{R}^{d}$, which are aggregated to form matrices $E_{a}\in\mathbb{R}^{N\times d}$ and $E_{t}\in\mathbb{R}^{N\times d}$.  The similarity matrix C in the CLAP framework is given by:

$$C=\tau(E_{t}\cdot E_{a}^{\top})$$ 

The CLAP loss minimizes the discrepancy between audio and text representations, is denoted by L and formulated as:

$$\mathcal{L}=0.5(l_{text}(C)+l_{audio}(C))$$ 

Here, $\tau$ serves as a scaling factor, modulating the effect of the similarity scores.  While the original CLAP framework is valuable for audio-text pretraining, it does not allow causality expression.  Addressing this gap, we introduce a counterfactual natural language strategy to infuse the CLAP framework with causality. 

### Causal Identification and Intervention
The counterfactual sentences in our model are generated through a prompt-based intervention on an observed caption y, represented as $y^{*}=do(y|p)$.  This prompt p is designed to fulfill three key aspects: 
* It is factually grounded. 
* It identifies acoustic sources in captions to serve as identifying causes. 
* It manipulates these sources to alter the caption, to serve as causal interventions. 

Identifiability means that all causal sources of acoustic waves can be obtained only from the language.  Examples of these transformations are depicted in Table 1. 

| Dataset | Original Caption | Generated Counterfactual |
| :--- | :--- | :--- |
| Clotho | A gun is loaded, then loaded by hand some more | A piano is played, then played by hand some more. |
| | A few gunshots are fired at the target shooting range | A few fireworks light up the night sky at shooting range. |
| | An adult male speaks and a crash occurs | An adult male speaks and a thunderstorm rumbles. |
| AudioCaps | Large group of people clapping | Flock of birds chirping in unison. |
| | Idling car, train blows horn and passes | Dogs barking, train blows horn and passes. |
| | A crowd of people indoors talking | A group of cars honking on a busy street. |
| MACS | Adults and children are walking and talking | Cars and trucks are honking and zooming. |
| | Adults talking and some footsteps coming across | Dogs barking and some footsteps coming across. |
> *Table 1: Samples of original captions from the Clotho, MACS, and AudioCaps datasets and their counterfactual pairs.* 

### Control Mechanisms for Counterfactual Language
We utilize prompts based on the Chain-of-Thought (CoT) method to align with objectives of causal identification and intervention and generating counterfactuals.  We introduce a two-step prompting mechanism, denoted as $p=\{p_{1},p_{2}\}$.  In this mechanism, $p_{1}$ anchors the discussion in factual elements, dissects acoustic objects, and plays the role of causal identifier.  Concurrently, $p_{2}$ governs the generation of counterfactual statements by intervening in the identified causal acoustic sources.  The counterfactuals can range from full negative examples, as found in hard negative sampling, to minor, physically plausible counterfactual scenarios. 

### Loss Functions to Incorporate Counterfactuals
The angle loss aims to minimize the angular difference between factual and counterfactual captions.  To encourage factual consistency between audio samples and their corresponding captions, we define the factual consistency loss.  The total loss, combining both the factual consistency loss and angle loss, is expressed as:

$$L_{total} = w_1 L_{angle} + w_2 L_{factual\_consistency}$$ 

Choices of hyperparameters are pragmatic, serving as a tradeoff to best exploit the capabilities of counterfactuals while ensuring factual consistency. 

---

## 3. EXPERIMENTAL DESIGN

### 3.1. Encoders
* **Audio encoder:** We use PANNs encoder, specifically ResNet-38 has been used with pretrained weights loaded with adapter layers to fine-tune model and align the embeddings. 
* **Text encoders:** We use the same CLIP text encoder modules provided from HuggingFace for encoding captions and counterfactuals.  The weights of the encoders were frozen in all stages. 

We employed logarithmic Mel spectrograms of audio sampled at 32kHz.  The hop size is set to 320 frames, the window size to 1024 frames, and we utilized 64 Mel bins spanning the frequency range of 50-14000 Hz.  Audio clips were randomly truncated to contiguous 10-second segments for training purposes, with zero padding applied for shorter clips.  The captions remained unaltered. 

### 3.2. Data
* **Training datasets:** Total of 44,292 from AudioCaps, 29,646 pairs from Clotho (each audio has five captions, so we created five pairs per clip), and 17,276 pairs from MACS have been used during pretraining. 
* **Test datasets:** We use the Clotho dataset for evaluating the model's performance in language-based audio retrieval task.  For evaluating the model's performance as zero-shot classification in conventional problems with limited classes, we use the Environmental Sound Classification 50 (ESC-50) with 50 predefined categories related to audio classes, and UrbanSound8K (US8K) with ten classes. 

### 3.3. Baseline
We adopt the approach from CLAP, and train our version with the same datasets we used in generating counterfactuals, including only AudioCaps, Clotho, and MACS. 

---

## 4. RESULTS AND DISCUSSIONS

### 4.1. Evaluation on Downstream Tasks
**Results on Clotho:** We use Clotho to test the performance of our method on language-based audio retrieval task.  As listed in Table 2, our method's performance on text-based retrieval tasks yields a 43% improvement in top-1 accuracy, reinforcing its superior precision.  Notably, the performance has slight improvement for top-10 retrieval tasks.  

| Method | Top-1 | Top-10 |
| :--- | :--- | :--- |
| CLAP | 0.088 | 0.395 |
| Our method | 0.126 | 0.423 |
> *Table 2: Performance (out of 1) on the Clotho evaluation set for text-to-audio retrieval is listed.* 

**Results on ESC-50 and US8K:** We evaluated the zero-shot classification performance of our proposed model on two benchmark datasets, ESC-50 and US8K, and summarized in Table 3.  As the table shows, our model performs commendably on the ESC-50 dataset, which features many classes.  This performance is slightly better than that of the CLAP method.  Conversely, the performance lags on the US8K dataset when compared to CLAP.  One reason might be that the number of classes in US8K is much lower, in contrast to what our model learned during the training.  

| Method | ESC-50 | US8K |
| :--- | :--- | :--- |
| Wav2CLIP | 0.414 | 0.404 |
| AudioClip | 0.694 | 0.653 |
| CLAP | 0.729 | 0.798 |
| Our method | 0.744 | 0.475 |
> *Table 3: Comparison of zero-shot performance between our proposed method and existing methods on ESC-50 and US8K.* 

### 4.2. Ablation Studies
Evidently, starting from a random guess, the embeddings evolve with each subsequent addition of the loss terms.  One particular observation is that audio embeddings of the CLAP are closer to the fact than counterfactuals.  This observation, by itself, shows that CLAP is favorably learning to stay closer to facts.  By incorporating counterfactuals via angle loss, we observe that the audio embeddings of our method get distant from the counterfactuals, staying closer to the facts.  Alternatively, by only including factual loss, audio embeddings tend to align fully with factual embeddings, staying closer to the facts. 

| $w_1$ | $w_2$ | Top-1 | Top-10 | $sim(x,y)>sim(x,y^*)$ |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0 | 0.0002 | 0.01 | $517.80\pm237.36$ |
| 1 | 0 | 0.0819 | 0.3365 | $1000.80\pm5.76$ |
| 0 | 100 | 0.1328 | 0.4379 | $861.80\pm293.76$ |
| 1 | 100 | 0.1102 | 0.3782 | $967.80\pm43.36$ |
> *Table 4: Ablations on different combinations of $w_1$ and $w_2$.* 

In Table 4, adding angle loss $w_1$ and using counterfactuals improve our audio embeddings.  Another pattern is that having factual consistency loss improves accuracy; however, the closeness of audio-text, as compared to counterfactuals, is not as good as when we add angle loss.  Therefore, there is a trade-off between facts and counterfactuals.  In contrast, adding angle loss incorporates counterfactuals in training, helping to distinguish between descriptions that might be directly (as facts/captions) or indirectly (as counterfactual captions) related to the same audio. 

---

## 5. CONCLUSION AND FUTURE DIRECTION
For the first time, we incorporate a counterfactual framework into the audio domain.  We leverage LLMs for counterfactual reasoning by prompt-based intervention on the identified acoustic objects.  This integration aims to identify variations in audio-text representations by focusing on natural language as a surrogate for the lack of alternative audio-text data when acoustic waves' origin and root cause vary.  Our method exploits human-generated reference captions for surrogate counterfactuals and adopts them to the audio-text pretraining in a triplet model with factual consistency.  

Counterfactual natural language effectively compensates for the scarcity of comprehensive counterfactual audio data for ethical or feasibility reasons and enhances the distinguishability of audio-text models.  Empirical evaluations using the various datasets substantiated the effectiveness of our method.  In particular, our method yields a 43% improvement in top-1 accuracy for open-text tasks.  Future research may explore the efficacy of these counterfactuals in challenging existing factual representations and their subsequent impact on audio-text correlation.  Another avenue for future work could involve examining various levels of counterfactual reasoning. 