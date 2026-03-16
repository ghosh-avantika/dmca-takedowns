import torch

data = torch.load('dataset_splits_WITH_PSEUDO_LABELS.pt')

train_X = data['train']['embeddings']  # [165, 512]
train_y = data['train']['labels']       # [165]

val_X = data['val']['embeddings']       # [35, 512]
val_y = data['val']['labels']           # [35]

test_X = data['test']['embeddings']     # [36, 512]
test_y = data['test']['labels']         # [36]