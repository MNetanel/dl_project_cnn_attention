**Attention Mechanisms for Image Classification**

*Netanel Madmoni & Gil Ben Or*

---

This repo contains our final project for the Intro to Deep Learning Course @ the department of industrial engineering, Ben-Gurion University of the Negev.

The attention mechanism is used in various field of deep learning to adjust the weights of certain parts of an input based on their importance for the task at hand. The usage of the attention mechanism in the field of computer vision has gained popularity in recent years, and attention modules are being used - either on their own or in conjunction with a convolutional neural network -  for various computer vision tasks, such as image classification and object detection.

We propose a spatial attention module based on the architecture of an autoencoder, which compresses an input to a small spatial dimension and then attempts to restore it. The idea is to re-purpose the autoencoder architecture to keep certain parts of an image that are important for the image classification task. We also explore the combination of various attention modules in order to enhance their performance in the classification task.
<p align="center">
<img src="https://github.com/MNetanel/dl_project_cnn_attention/assets/20209534/5d90ddec-14d0-4dcc-8101-15acf0f89a87" width=70%>
</p>

By using only spatial attention, the CNN that had our autoencoder-inspired attention block achieved similar results on the CIFAR10 dataset to the netwotks that incorporated channel-based attention in them. It also achieved similar results to a deeper CNN while having about 17\% less parameters. We believe that while not showing significantly better results, our work demonstrate the potential of our novel attention block.

