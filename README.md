# Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning (2019)
# Paper: https://arxiv.org/abs/1908.02983

Official Pytorch implementation from authors. Our pseudo-labeling approach achieves state-of-the-art performance for semi-supervised learning (SSL) in Image Classification. You can find an example script to run the poroposed SSL approach on CIFAR-10 with 1000 labeled samples in [RunScripts_SOTA1000.sh](https://github.com/EricArazo/PseudoLabeling/cifar10/RunScripts_SOTA1000.sh) and for CIFAR-100 with 4000 labeled samples in [RunScripts_SOTA4000.sh](https://github.com/EricArazo/PseudoLabeling/cifar100/RunScripts_SOTA4000.sh). Execute the script from the corresponding folder to train the model.

 | Dependencies  |
| ------------- |
| python == 3.5.2     |
| pytorch == 0.4.1     |
| cuda == 8.0|
| torchvision == 0.2.1|
| matplotlib == 3.0.1|
| scikit-learn == 0.20.0|
| tqdm == 4.28.1|
| numpy == 1.15.3|


### Parameters details
Execute the following to get details about parameters. Most of them are set by default to replicate our experiments.
``` sh
$ python train.py --h
```
The most relevant parameters are the following:
* --labeled_samples: Number of labeled samples 
* --epoch: Number of epochs of training
* --M: Epochs where the learning rate is divided by 10
* --label_noise: ratio of unlaebeled samples to be relabeled with a uniform distribution

To run the experiments download the corresponding dataset in the folder ./CIFAR10/data or ./CIFAR100/data.

### Accuracies

|Number of labeled samples |500|1000|2000|4000|10000|
|----|----|----|----|----|----|
|CIFAR-10|14.07 ± 0.49|12.63 ± 0.54|9.21 ± 0.58|7.09 ± 0.14|----|
|CIFAR-100|----|----|----|39.67 ± 0.13|31.00 ± 0.25|


### Acknowledgements

We would like to thank [1] (https://github.com/benathi/fastswa-semi-sup) for the "13-layer network" implmentation and [2] (https://github.com/CuriousAI/mean-teacher) for the data sampler code that we use in our code.

[1] Athiwaratkun, Ben and Finzi, Marc and Izmailov, Pavel and Wilson, Andrew Gordon, "There Are Many Consistent Explanations of Unlabeled Data: Why You Should Average", in International Conference on Learning Representations (ICLR), 2019

[2] Antti Tarvainen, Harri Valpola, "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results", in Advances in neural information processing systems, 2017.  


### Please consider citing the following paper if you find this work useful for your research.

```
 @article{PseudoLabel2019,
  title = {Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning},
  authors = {Eric Arazo and Diego Ortego and Paul Albert and Noel E O'Connor and Kevin McGuinness},
  journal = {arXiv:1908.02983},
  month = {August},
  year = {2019}
 }
```

Eric Arazo, Diego Ortego, Paul Albert, Noel E. O'Connor, Kevin McGuinness, Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning, arXiv:1908.02983, 2019
