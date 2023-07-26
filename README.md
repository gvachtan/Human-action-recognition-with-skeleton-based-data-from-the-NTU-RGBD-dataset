# Human-action-recognition-with-skeleton-based-data-from-the-NTU-RGBD-dataset
A code for human action recognition using skeleton data from the NTU RGBD dataset, using a multihead attention GCA LSTM model, inspired from the following paper: https://arxiv.org/abs/1707.05740.
This particular task has two evaluation protocols, a cross-subject and cross-view evaluation, I use cross-view evaluation.
Before running the code you need to have your data in a numpy format. Initially, the data are in a ".skeleton" format. You should convert them into ".txt" form and then you convert the data from ".txt" into "numpy" by using the code in the following repo: https://github.com/shahroudy/NTURGB-D/tree/master/Python.
It achieves a validation accuracy up to 80%.
If someone takes the code and manages to improve it's performance, I'd be happy to be informed about it.
