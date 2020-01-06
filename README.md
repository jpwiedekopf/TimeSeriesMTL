# TimeSeriesMTL
Code for my Master's Thesis at the Institute of Medical Informatics, Universität zu Lübeck.

I carried out studies on Multi-Task Learning using three approaches (Hard
Parameter Sharing, [Regularised Soft Parameter
Sharing](https://arxiv.org/abs/1606.04038) and [Cross-Stitch
Networks](https://arxiv.org/abs/1604.03539)) on two time-series datasets with
sensor-based data
([OPPORTUNITY for Human Activity
Recognition](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition) 
and [DEAP for Emotion Recognition from EEG](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).

For a gentle introduction into Multi-Task Learning, I recommend this [excellent
paper by S. Ruder](https://arxiv.org/abs/1706.05098).

Here is the abstract from my thesis:

> Multi-Task Learning (MTL) in the domain of Deep Neural Networks (DNNs) is an
> idea where a network performs multiple tasks at once to produce latent representations
> of the input that are more plausible than what is generated using classical Single-Task
> Learning (STL). In this way, many approaches to this concept have demonstrated
> results that surpass those using STL. However, the use of MTL approaches on sensor-
> based time-series data has not received much attention so far. This is unfortunate, as
> time-series data processing has an enormous range of potential applications, especially
> in the medical domain.
>
> In this thesis, three very different approaches to MTL—Hard Parameter Sharing (HPS),
> Soft Parameter Sharing (SPS) and Cross-Stitch Networks (CSNs)—are applied to two
> different datasets from the domain of Human Activity Recognition (HAR) and Emotion
> Recognition—OPPORTUNITY and DEAP. We demonstrate that not every approach
> is equally beneficial. In particular, benefits were observed using HPS on both datasets
> and CSNs on only one dataset. We could not demonstrate that the chosen approach
> to SPS works for time-series. To aid further research in this field, our source code is
> made public. 

