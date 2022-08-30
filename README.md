# Task-Oriented-Semantic-Communications

This is the implementation of *Intelligent task-oriented semantic communication method in artificial intelligence of things*<br />

## Introduction of the main files
1 “ResNet18_CAM.py” is used to generate semantic importance ranking<br />
2 “mi_estimators.py” is used to estimate mutual information<br />
3 “MI+Classification_resnet.py” and “resnet_10_classes_channel.py” are the training code for considering and not considering mutual information between the channel input and output in the loss, respectively<br />
4 “classification_crop.py” is used to compress semantics<br />
5 “classification_test_resnet.py” is the test code<br />


## Bibtex
article{liu2021sc,
  author={C. Liu and C. Guo and Y. Yang and C. Feng and Q. Sun and J. Chen},
  journal={Journal on Communications}, 
  title={Intelligent task-oriented semantic communication method in artificial intelligence of things}, 
  year={2021},
  volume={42},
  number={11},
  page={97-108}
  
## Notes
“resnet18-5c106cde.pth” is the inital parameters of Resnet18 which can be download online<br />
The proposed method can be generlized to other network structures
