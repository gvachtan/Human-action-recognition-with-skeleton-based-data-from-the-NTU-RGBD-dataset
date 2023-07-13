# Human-action-recognition-with-skeleton-based-data-from-the-NTU-RGBD-dataset
A code for human action recognition using skeleton data from the NTU RGBD dataset, using a multihead attention GCA LSTM model, inspired from the following paper: https://arxiv.org/abs/1707.05740.
This particular task has two evaluation protocols, a cross-subject and cross-view evaluation, I use cross-view evaluation.
Before running the code you need to have your data in a numpy format, you can find code and instructions in the following repo: https://github.com/shahroudy/NTURGB-D
It achieves a validation accuracy up to 80%.
If someone takes the code and manages to improve it's performance, I'd be happy to hear from you about it.
