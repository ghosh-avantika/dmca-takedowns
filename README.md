# Between Inspiration and Imitation:

## Graded Fashion Infringement Assessment with Deep Metric Learning and a Calibrated Decision Layer
https://drive.google.com/file/d/1u6ctHFmK-Yk9jzx8in-LNuXBT2PY_blu/view?usp=sharing

---

## Abstract

Fashion design infringement sits at an ambiguous boundary between lawful inspiration and illegal copyright, making it poorly suited for binary visual similarity systems. This study formulates fashion infringement assessment as a graded three-class classification task over *original*, *similar*, and *knockoff* cases.

The implemented system follows a two-stage pipeline:

1. a deep metric-learning component provides fashion-aware visual representations, and
2. a calibrated downstream decision layer maps pairwise embedding-derived features into interpretable infringement categories and DMCA-style actions.

Using a curated dataset of fashion design cases, cosine-threshold baselines are compared with learned classifiers operating on nearest-original comparison features. The strongest current result is a pairwise multilayer perceptron decision layer, which achieves a test macro-F1 of **0.7368** in the uncalibrated comparison setting. After temperature scaling, the calibrated deployment-oriented version achieves **0.6926** test macro-F1 while improving validation expected calibration error from **0.1042** to **0.0857**.

These results indicate that graded infringement assessment is better handled by a calibrated non-linear decision layer than by fixed similarity thresholds alone, and that separating representation learning from decision calibration provides a more interpretable and practically useful framework for computer-assisted fashion copyright review.

---

## Introduction

Fashion design infringement is difficult to assess because the domain sits between artistic expression, functional design, and rapid commercial imitation. Many fashion copyright cases do not involve exact duplication, but rather ambiguous overlap in motifs, prints, proportions, or detailing. This creates a poor fit for binary knockoff/original systems. Legal analysis of fashion copyright has also emphasized that many design elements, including silhouettes, trends, and broad stylistic conventions, are only weakly protected or treated as functional rather than expressive, making the boundary between lawful inspiration and infringement especially difficult to prove and prosecute accordingly [17]. In the UK, the procedural reality of such lower-value and less complex intellectual property disputes is also reflected in the specialist role of the Intellectual Property Enterprise Court [14].

This study investigates whether fashion infringement can be modeled more effectively as a graded decision problem. Rather than asking only whether two designs are originals or knockoffs, the study considers three categories:

* *Original*
* *Similar*
* *Knockoff*

This framing reflects the practical reality that many fashion cases occupy an intermediate region of stylistic similarity that is neither clearly lawful inspiration nor clearly illegal copying.

This study is a two-stage pipeline that separates visual representation learning from downstream decision-making. First, a deep triplet metric-learning stage based on a ResNet encoder is used to learn fashion-aware embeddings from image data, as seen in prior metric-learning literature on similarity-based representation spaces [1, 2, 3, 16]. Second, a calibrated decision layer takes pairwise features derived from a candidate design and its nearest *original* reference and converts them into both class predictions and confidence-aware policy actions.

At the prediction level, the model assigns one of the three labels *original*, *similar*, or *knockoff*. At the policy level, those calibrated probabilities are translated into operational outcomes:

* **auto_flagged** for high-confidence predicted knockoffs
* **review** for borderline or uncertain cases
* **no_action** for low-confidence non-critical cases

This differs from prior retrieval-style systems by focusing not only on similarity estimation, but on interpretable and calibrated decision support for ambiguous cases in which the main question is not simply *“what is most similar?”*, but *“what action should be taken given the model’s confidence?”*

### Research Questions

1. Can embedding-based visual similarity support reliable distinction between *original*, *similar*, and *knockoff* fashion cases?
2. How sensitive are graded infringement decisions to the choice of thresholding versus learned non-linear decision layers?
3. Can deep metric learning on fashion imagery provide a stronger representation space for downstream infringement assessment?

The strongest current evidence comes from the multilayer perceptron calibrated decision layer rather than from a completed end-to-end evaluation of metric-learning variants. The experiments show that decision calibration and pairwise feature design are critical to the task, and that they provide a strong baseline against which future representation-learning improvements can be measured.

Compared with the CW3 plan, the emphasis of the final report shifts slightly. The interim plan was to explore a broad comparison of metric-learning variants and downstream effects, whereas the classification results are strongest for the calibrated multilayer perceptron decision layer built on the present embedding pipeline. The final report reflects this change directly. The study treats the deep metric-learning component as the representation-learning foundation of the project, while presenting the most complete empirical findings around graded classification, calibration, and DMCA-style triage.

