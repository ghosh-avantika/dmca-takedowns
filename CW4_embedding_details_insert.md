Embedding details to insert into the report

Dataset section replacement paragraph

The custom infringement dataset is represented using fixed-length visual embeddings rather than raw pixels. Each fashion image is first mapped to a 512-dimensional feature vector, and each pairwise sample is formed from the embeddings of the compared designs together with its label (`original`, `similar`, or `knockoff`). These precomputed vectors are stored in the train, validation, and test split files used throughout the decision-layer experiments. In the current pipeline, the embeddings used for downstream classification are treated as the representation-learning stage, while the infringement classifier operates only on derived pairwise features. This allows the same train/validation/test split to be reused when comparing threshold baselines, logistic regression, and multilayer perceptron decision layers under identical evaluation conditions.

Dataset section add-on for DeepFashion

For representation learning, the project uses DeepFashion as an auxiliary image dataset because it provides item identity information even though it does not contain infringement labels. These identity annotations allow positive pairs or triplets to be constructed from different views of the same clothing item and negatives to be sampled from different items. The resulting model is therefore trained to organize fashion images by visual similarity before any infringement-specific supervision is applied. This separation is important because the large fashion dataset provides generic garment-level visual structure, while the smaller infringement dataset is reserved for the higher-level decision task.

Methodology section replacement paragraph

The final system follows a two-stage pipeline. First, a ResNet-based metric-learning model is used to produce normalized embeddings for fashion images. In the implemented triplet-learning setup, the encoder is initialized with ImageNet-pretrained weights and fine-tuned on DeepFashion using anchor-positive-negative triplets derived from item identities. Fine-tuning is restricted to the later layers of the network to preserve general visual features while adapting the representation to fashion-specific cues such as silhouette, texture, and detailing. Second, infringement prediction is performed by a separate decision layer operating on pairwise embedding features rather than on raw images directly.

Methodology section add-on for pairwise features

Given an input design embedding, the decision layer compares it to the nearest `original` reference embedding in the training set and constructs pairwise features from that comparison. These features include cosine similarity, cosine distance, Euclidean distance, summary statistics of the absolute embedding difference, and the element-wise absolute difference vector itself. Threshold baselines use the distance features directly to define partitions for `original`, `similar`, and `knockoff`, while the learned MLP decision layer uses the richer pairwise feature representation to model non-linear class boundaries. This formulation makes the model outputs easier to interpret than a direct end-to-end image classifier because the final decision can be related back to explicit similarity measurements in embedding space.

Safe support sentence for the representation-learning claim

Rather than claiming that a two-stage pipeline is universally superior to end-to-end multiclass classification, this project uses the separation between representation learning and decision calibration to improve interpretability and to make threshold sensitivity, calibration behaviour, and failure cases directly measurable. Empirically, the learned decision layers built on embedding-derived pairwise features substantially outperform simple cosine-threshold baselines on the three-class infringement task, which supports the usefulness of this decomposition for the present setting.

Short implementation note you can cite in prose

In the implemented codebase, the metric-learning stage is handled by the triplet ResNet training pipeline, and the downstream infringement classifier is implemented as a separate decision layer that consumes pairwise features derived from the learned embeddings.
