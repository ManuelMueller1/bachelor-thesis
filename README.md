The directory "ControllableModels" contains the implementation of the developed ControllableReLiNet models. 
For time effieciency during training, there exist 2 different implementations of the SVD method. One script vectorizes most of the calculations, just like the other two ControllableReLiNet methods.
As the Gram-Schmidt process is iterative, the other script uses only iterative calculations. The training of this method on a CPU was faster than the training of the SVD method on a GPU.

The code for the Gram-Schmidt-orthogonalisation is adaped from https://github.com/legendongary/pytorch-gram-schmidt .
