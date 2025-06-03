# RotationForest
A python implementation of the Rotation Forest algorithm per https://arxiv.org/abs/1809.06705. 

# What is Rotation Forest?
Rotation Forest is an ensemble classification method similar to Random Forest, which addresses one of Random Forest's bigger weaknesses—its component models, decision trees, can only partition the feature space orthogonally (perpendicular to the feature axes). To represent non-orthogonal boundaries, decision trees require a large number of splits, which can lead to overfitting and poor performance. Random Forest, which predicts the most common output of n trees trained on bootstrap samples, still may run into issues because of the base trees' gridlike decision structure. 

Rotation Forest tries to mitigate this by randomly "rotating" each tree in the forest. In practice, this is done by randomly splitting up the features of the model into some number of partitions, sampling from these partitions, and applying PCA on the samples to get a "rotation matrix" (the eigenvector matrix, or principal components, depending on your background). By multiplying the original feature partitions and the rotation matrix, we rotate the features of the model, with randomness provided by the random sampling and random partitions. This means that each tree in the forest is "pointing in a different direction" in the feature space, allowing the ensemble to approximate curves more effectively than Random Forest.

# Example: Quadratic Data + noise in other dimensions

![Train Data visualized](/mean0std1/Figure_1.png)

Here's a simple toy dataset with two quadratic curves + noise. We'll try fitting a Random Forest with stock settings and 200 trees, and a Rotation Forest with stock settings and 200 trees.

![Random Forest Predictions](/mean0std1/Rand.png)

The Random Forest gets a bit confused in the middle due to the non-gridlike boundary between the classes.

![Rotation Forest Predictions](/mean0std1/Rot.png)

Rotation Forest, while not perfect, learns the decision boundary much more accurately.
