# ContrastiveLearning

In this project, I have used three different contrastive learning algorithms to produce
an informative embedding of the Fashion MNIST dataset. Each algorithm uses an open-source augmentation module 
from the PyTorch library and a simple convolutional neural network embedding scheme. The first two algorithms are applied in the 
self-supervised setting, and the third algorithm is applied in the supervised setting. I measure the quality of each embedding
through a series of downstream classification and clustering tasks.

The results of the first algorithm, triplet loss, is shown in the ```self_triplet.ipynb``` notebook. The results of the second
algorithm, self-supervised contrastive loss, is shown in the ```self_supcon.ipynb``` notebook. The results of the final algorithm,
supervised contrastive loss, is shown in the ```supcon.ipynb``` notebook. The first algorithm is specified in 
_FaceNet: A Unified Embedding for Face Recognition and Clustering_ by Florian Schroff, Dmitry Kalenichenko, and James Philbin.
The other two algorithms are specified in _Supervised Contrastive Learning_ by Khosla et al.

A detailed written report specifying each of my experiments can be found within the ```Report``` directory of this repo. The ```Report```
directory also includes a brief 10 minute presentation of my motivation and conclusions from this project.
