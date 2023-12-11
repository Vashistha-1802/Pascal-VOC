# Pascal-VOC
The main goal of this challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e. not pre-segmented objects). It is fundamentally a supervised learning learning problem in that a training set of labelled images is provided. Twenty object classes that have been selected 

# Problem Statement
The objective is to identify objects from a diverse range of visual classes within realistic scenes. The dataset comprises 20 classes, including animals, vehicles, furniture, and more. The classification task involves predicting the presence or absence of each class in a given image.

#Data
Utilizing the Pascal VOC 2012 dataset via PyTorch, the project focuses on the training and validation sets. Annotations include object class, bounding box, view, difficulty level, truncation, and occlusion. 'Difficult' marked objects are treated as negative examples.

#Loss Function
For the multi-label classification task, binary cross-entropy with logits loss is employed. The loss function, implemented in PyTorch as torch.nn.BCEWithLogitsLoss(), offers numerical stability over the sigmoid-binary cross-entropy sequence.

#Metrics
Average precision serves as the performance metric, representing the average of maximum precisions at various recall values. Notably, accuracy is deemed inadequate due to the nature of the problem, where a single label may suffice for a complex image.

#Model
ResNet50 is chosen as the deep learning architecture for its memory efficiency. Transfer learning is applied, leveraging similarities between object classes and ImageNet classes.
