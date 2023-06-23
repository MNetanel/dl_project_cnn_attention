# SE

## Explanation
https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7

Diagram by ChatGPT:
```mermaid
graph TD
    A["Input Feature Map (H x W x C)"]
    B["Global Average Pooling (1 x 1 x C)"]
    C["Fully Connected Layer (1 x 1 x C/r)"]
    D["ReLU Activation (1 x 1 x C/r)"]
    E["Fully Connected Layer (1 x 1 x C)"]
    F["Sigmoid Activation (1 x 1 x C)"]
    G["Output Feature Map (H x W x C)"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G

```


## Ideas
### Squeeze

| Parameter       | Tested Values            | Selected Value | To Try                        |
|:--------------- |:------------------------ |:-------------- |:----------------------------- |
| Reduction ratio | 2, 4, 8, 16, 32          | 16             |                               |
| Operation       | max pooling, avg pooling | avg pooling    | More sophisticated techniques |

### Excitation

### General

| Parameter     | Tested Values       | Selected Value | To Try |
|:------------- |:------------------- |:-------------- |:------ |
| Non linearity | ReLU, Tanh, Sigmoid | Sigmoid        |        |




# CBAM

## Explanation

https://medium.com/visionwizard/understanding-attention-modules-cbam-and-bam-a-quick-read-ca8678d1c671

Diagram by ChatGPT:

```mermaid
graph TD
    A["Input Feature Map (H x W x C)"]
    B["Channel Attention (1 x 1 x C)"]
    C["Spatial Attention (H x W x 1)"]
    D["Element-wise Multiplication"]
    E["Output Feature Map (H x W x C)"]

    A --> B
    B --> C
    C --> D
    A --> D
    D --> E

```

## Ideas
### Channel attention
  * MLP
    * Number of hidden layers (1)
    * Size of hidden layer(s) or reduction ratio C/r (r=16)
    * max pooled + avg pooled feature maps. can we add / replace them with other methods?
    * Weights are shared for both max pooled and avg pooled feature maps

### Spatial attention
  * Generation - avg&max pooling better than 1x1 conv
  * filter size (7) - they found out it is better than smaller ones
* Arrangement - they found out best is channel → spatial
* Merging the two modules using element-wise summation, no learnable parameters


# BAM

## Explanation
https://medium.com/visionwizard/understanding-attention-modules-cbam-and-bam-a-quick-read-ca8678d1c671

Diagram by ChatGPT:
```mermaid
graph TD
    A["Input Feature Map (H x W x C)"]
    B["Channel Attention (1 x 1 x C)"]
    C["Spatial Attention (H x W x 1)"]
    D["Element-wise Multiplication"]
    E["Output Feature Map (H x W x C)"]

    A --> B
    A --> C
    B --> D
    C --> D
    D --> E

```

## Ideas

# ECA
> Explanation: https://blog.paperspace.com/attention-mechanisms-in-computer-vision-ecanet/

Diagram by ChatGPT:
```mermaid
graph TD
    A["Input Feature Map (H x W x C)"]
    B["Convolutional Layer (H x W x 1)"]
    C["Channel Attention (H x W x 1)"]
    D["Element-wise Addition"]
    E["Output Feature Map (H x W x C)"]

    A --> B
    B --> C
    A --> D
    C --> D
    D --> E

```


## Ideas

# General Ideas

* Architetcures to put the mosules in
* where in the network to put them
* combining modules, repeating modules
* usage for different tasks
