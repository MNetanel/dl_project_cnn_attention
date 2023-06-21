# CBAM

* Channel attention
  * MLP
    * Number of hidden layers (1)
    * Size of hidden layer(s) or reduction ratio C/r (r=16)
    * max pooled + avg pooled feature maps. can we add / replace them with other methods?
    * Weights are shared for both max pooled and avg pooled feature maps
* Spatial attention
  * Generation - avg&max pooling better than 1x1 conv
  * filter size (7) - they found out it is better than smaller ones
* Arrangement - they found out best is channel → spatial
* Merging the two modules using element-wise summation, no learnable parameters

‌

# General

* Architetcures to put the mosules in
* where in the network to put them
* combining modules, repeating modules
* usage for different tasks
