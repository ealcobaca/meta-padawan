# Meta-Padawan


## The Competition
The [MetaDL](https://metalearning.chalearn.org/) competition brings us several challenges. But the most difficult to address, in our opinion, was those related to the limitation of time and data.
So the solution should consider that there is not enough time to train a big model. We have only two hours per dataset.
Imagine tuning the hyperparameters if there is no time to train a large model. No way!
Lastly, we had only a few examples of each class in the dataset. Creating a model that generalizes well with limited data is very challenging.

## Our solution
We hypothesized that we can meta-learn from previously trained models.
Then, instead of creating a new deep learning model, we can faster generate new models with high generalization capacity by combining different features descriptors learned from pre-trained models in other datasets and simple features learned from the new images at hand.

## Relevant points
* The solution is a simple approach that can be easily implemented since there is no need for high deep learning domain knowledge nor the ability to design robust architectures
* When compared with large architectures, fewer parameters need to be optimized during the learning phase
* There is no need for expensive hyperparameter optimization steps. We propose the use of a simple linear model
* The meta-learning takes place using features descriptors learned in different pre-trained models (InceptionResNetV2 and VGG12) combined with features from the original images (PCA-based)
* A negative point is that the combination of several pre-trained models can lead to the curse of dimensionality that should be treated in some way


## News!
* You can see our full presentation on [YouTube](https://www.youtube.com/watch?v=XJiT_dbgvQs&ab_channel=EdesioAlcoba%C3%A7a). 
* We got the 3rd place 	:champagne: :confetti_ball: :tada:

## How to use
We can find our proposed method in folder `meta-padawan`.
Information about the conpetition and how to run this experiments, baselines and our proposed method can be found [here](https://github.com/ealcobaca/metadl/blob/master/README2.md).
