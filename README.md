# Machine Learning Concepts

A comprehensive guide to understanding machine learning concepts, algorithms, and techniques.

## Table of Contents

* [Supervised Learning](#supervised-learning)
* [k-NN Method](#k-nn-method)
* [Decision Trees](#decision-trees)
* [Basic Parametric Models](#basic-parametric-models)
* [Model Evaluation & Performance](#model-evaluation--performance)
* [Learning Parametric Models](#learning-parametric-models)
* [Ensemble Methods](#ensemble-methods)
* [Neural Networks & Deep Learning](#neural-networks--deep-learning)
* [Non-linear Transformations & Kernels](#non-linear-transformations--kernels)
* [Bayesian Methods](#bayesian-methods)
* [Generative & Unsupervised Learning](#generative--unsupervised-learning)

## Supervised Learning

Supervised machine learning is a fundamental approach where models learn from labeled data to make predictions or decisions without explicit programming.

### Key Points

* **Labeled Data**: Algorithm learns from input-output pairs, with each input associated with a known output/label
* **Training Process**: Model adjusts parameters to minimize the difference between predictions and actual labels
* **Generalization**: Creates models that accurately predict outcomes for new, unseen data
* **Types**: Includes classification (discrete categories) and regression (continuous values)

### Learning Process

1. **Data Collection**: Gather dataset with features (inputs) and corresponding labels (outputs)
2. **Data Preprocessing**: Clean and prepare data, handle missing values, normalize features
3. **Model Selection**: Choose appropriate algorithm based on problem type and data characteristics
4. **Training**: Feed data into model to learn relationships between inputs and outputs
5. **Evaluation**: Assess model performance using metrics like accuracy, precision, recall
6. **Optimization**: Fine-tune model using hyperparameter adjustment and cross-validation
7. **Prediction**: Deploy trained model to make predictions on new data

### Resources

* [TensorFlow Playground](https://playground.tensorflow.org/) - Interactive neural network visualization
* [Supervised vs Unsupervised Learning](https://www.youtube.com/watch?v=xtOg44r6dsE) - Simplilearn tutorial
* [Introduction to Supervised Learning](https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/)

## k-NN Method

The k-Nearest Neighbors algorithm is a versatile supervised learning method for both classification and regression tasks.

### Key Points

* **Non-parametric**: Makes no assumptions about data distribution
* **Instance-based**: Stores all training data for prediction
* **Lazy Learning**: No discriminative function learning, memorizes training dataset
* **Distance Metric**: Uses Euclidean, Manhattan, or Hamming distance
* **k Selection**: Number of neighbors significantly affects performance

### How k-NN Works

1. Choose number k of neighbors
2. Calculate distance between query instance and training samples
3. Sort distances to determine k-nearest neighbors
4. For classification: Use majority vote
5. For regression: Use average value

### Advantages

* Simple to understand and implement
* Works well with multi-class problems
* No assumptions about data
* Naturally handles outliers

### Disadvantages

* Computationally expensive for large datasets
* Sensitive to irrelevant features and data scale
* Requires feature scaling
* Memory-intensive

### Resources

* [k-NN Classification Visualization](https://towardsdatascience.com/visualizing-k-nearest-neighbor-classification-f929e5a51d2e)
* [StatQuest: k-NN Explained](https://www.youtube.com/watch?v=HVXime0nQeI)
* [k-NN Tutorial in Python](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn)

## Decision Trees

Decision Trees are versatile algorithms used for both classification and regression, creating predictions through a tree-like model of decisions.

### Key Points

* **Tree Structure**: Root node, internal nodes (decisions), branches (outcomes), leaf nodes (predictions)
* **Splitting Criteria**: Uses metrics like Gini impurity or information gain
* **Interpretability**: Provides clear visualization of decision process
* **Data Handling**: Works with both numerical and categorical data
* **Overfitting Risk**: Can create complex trees needing pruning

### How Decision Trees Work

1. Start at root node with entire dataset
2. Find best feature to split data using chosen criterion
3. Create child nodes for each split outcome
4. Repeat steps 2-3 for each child until stopping criterion met
5. Assign class labels or values to leaf nodes

### Advantages

* Easy to understand and interpret
* Requires little data preparation
* Handles both numerical and categorical data
* Works well with large datasets

### Disadvantages

* Can create overly complex trees
* Unstable to small data changes
* Biased towards dominant classes
* May overfit without pruning

### Resources

* [R2D3 Decision Tree Visualization](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
* [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
* [Decision Tree Algorithm Guide](https://towardsdatascience.com/decision-tree-algorithm-explained-83beb6e78ef4)



## Basic Parametric Models

### Linear Regression

Linear regression models the relationship between dependent and independent variables using a linear equation.

### Key Points

* **Linear Relationship**: Assumes linear relationship between input features and target variable
* **Continuous Output**: Predicts continuous output variable
* **OLS**: Uses Ordinary Least Squares to minimize squared residuals
* **Assumptions**: Includes linearity, independence, homoscedasticity, normality

### Mathematical Representation

For simple linear regression:
```
y = β₀ + β₁x + ε

Where:
- y is the dependent variable
- x is the independent variable
- β₀ is the y-intercept
- β₁ is the slope
- ε is the error term
```

### Advantages

* Simple to implement and interpret
* Computationally efficient
* Provides variable importance measure

### Disadvantages

* Assumes linear relationship
* Sensitive to outliers
* May underfit complex relationships

### Resources

* [Setosa's Linear Regression Visualization](http://setosa.io/ev/ordinary-least-squares-regression/)
* [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
* [Understanding Linear Regression](https://towardsdatascience.com/understanding-linear-regression-6a6a280ae75)

### Classification and Logistic Regression

Logistic regression predicts binary outcomes based on predictor variables.

### Key Points

* **Binary Output**: Used for binary classification problems
* **Sigmoid Function**: Uses logistic function for probability modeling
* **Decision Boundary**: Creates linear boundary between classes
* **Maximum Likelihood**: Uses maximum likelihood for parameter estimation

### Logistic Function

```
σ(z) = 1 / (1 + e^(-z))

Probability estimation:
P(Y=1|X) = 1 / (1 + e^(-(β₀ + β₁X₁ + ... + βₙXₙ)))
```

### Advantages

* Provides probability estimates
* Easily interpretable
* Computationally efficient
* Works well for linearly separable data

### Disadvantages

* Assumes linearity between variables and log-odds
* May underfit complex relationships
* Sensitive to outliers

### Resources

* [R2D3's Logistic Regression](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/)
* [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
* [Understanding Logistic Regression](https://towardsdatascience.com/understanding-logistic-regression-9b02c2aec102)

### Polynomial Regression and Regularization

Polynomial regression extends linear regression by including polynomial terms, while regularization prevents overfitting.

### Key Points

* **Non-linear Relationships**: Models curved relationships between variables
* **Degree Selection**: Polynomial degree affects model complexity
* **Regularization Types**: L1 (Lasso), L2 (Ridge), Elastic Net
* **Hyperparameter Tuning**: Regularization strength needs optimization

### Mathematical Representation

```
Polynomial: y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε

L1 (Lasso): Loss = MSE + λ∑|βᵢ|
L2 (Ridge): Loss = MSE + λ∑βᵢ²
```

### Resources

* [Polynomial Regression Visualization](https://towardsdatascience.com/polynomial-regression-bbe8b9d97491)
* [StatQuest: Ridge Regression](https://www.youtube.com/watch?v=Q81RR3yKn30)
* [Ridge and Lasso Tutorial](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/)

## Model Evaluation & Performance

### Expected New Data Error (Enew)

Enew measures how well a model generalizes to unseen data.

### Key Points

* **Generalization**: Measures model's ability to handle new data
* **True Performance**: Represents real-world application performance
* **Unobservable**: Must be estimated rather than directly calculated
* **Goal**: Minimizing Enew is often the ultimate aim in ML

### Mathematical Representation

```
Enew = Ex,y[L(y, f(x))]

Where:
- Ex,y is the expectation over true data distribution
- L is the loss function
- y is the true label
- f(x) is the model's prediction
```

### Resources

* [Bias-Variance Visualization](https://www.r2d3.us/visual-intro-to-machine-learning-part-2/)
* [Understanding Bias-Variance](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
* [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)



### Estimating Enew

Since Enew cannot be directly calculated, various methods are used to estimate it.

### Key Methods

* **Hold-out Validation**: Split data into training and validation sets
* **K-Fold Cross-Validation**: Divide data into K subsets, train on K-1, test on remaining
* **Leave-One-Out CV**: Special case where K equals number of samples
* **Bootstrap**: Create multiple datasets by sampling with replacement

### Advantages and Disadvantages

#### Hold-out
* **Pros**: Simple to implement
* **Cons**: May not be representative of full dataset

#### K-Fold CV
* **Pros**: More robust estimation
* **Cons**: Computationally intensive

#### LOOCV
* **Pros**: Uses all data efficiently
* **Cons**: Very computationally expensive

#### Bootstrap
* **Pros**: Works well with small datasets
* **Cons**: Can be biased

### Resources

* [Cross-Validation Visualization](https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/)
* [Cross Validation Tutorial](https://www.youtube.com/watch?v=fSytzGwwBVw)
* [k-fold Cross-Validation Guide](https://machinelearningmastery.com/k-fold-cross-validation/)

### Training Error–Generalisation Gap Decomposition

Understanding the relationship between training error and generalization ability.

### Key Components

* **Training Error**: Model's error on training data
* **Generalization Error**: Expected error on new data (Enew)
* **Generalization Gap**: Difference between generalization and training error

### Mathematical Representation

```
Generalization Error = Training Error + Generalization Gap
```

### Factors Affecting Gap

* **Model Complexity**: More complex models tend to have larger gaps
* **Dataset Size**: Larger datasets usually lead to smaller gaps
* **Regularization**: Proper regularization can reduce the gap

### Resources

* [Learning Curves Visualization](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
* [Overfitting and Underfitting](https://www.youtube.com/watch?v=SjQyLhQIXSM)
* [Understanding Generalization](https://towardsdatascience.com/understanding-the-generalization-gap-in-machine-learning-6a7a4a1a9a0a)

### Binary Classifier Evaluation Tools

Tools and metrics for assessing binary classification models.

### Key Metrics

#### Confusion Matrix
* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

#### Derived Metrics
* **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
* **Precision**: TP / (TP + FP)
* **Recall**: TP / (TP + FN)
* **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)

### Evaluation Curves

* **ROC Curve**: True Positive Rate vs False Positive Rate
* **Precision-Recall Curve**: Precision vs Recall
* **AUC**: Area Under the ROC Curve

### Resources

* [ROC Curve Visualization](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
* [ROC and AUC Explained](https://www.youtube.com/watch?v=4jRBRDbJemM)
* [Understanding AUC-ROC](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

## Learning Parametric Models

### Parametric Modeling Principles

Parametric models assume a specific functional form for the relationship between inputs and outputs.

### Key Principles

* **Fixed Function Form**: Model structure is predefined
* **Parameter Estimation**: Learning involves estimating best parameter values
* **Finite-Dimensional**: Number of parameters doesn't grow with data size
* **Assumptions**: Makes strong assumptions about data distribution

### Examples

* Linear Regression
* Logistic Regression
* Naive Bayes

### Resources

* [Parametric vs Non-parametric Models](https://www.youtube.com/watch?v=xZxNhwpwCnc)
* [Machine Learning Algorithms Guide](https://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/)

### Loss Functions and Likelihood Models

Loss functions quantify prediction errors while likelihood models provide a probabilistic framework for learning.

### Loss Functions

#### Key Concepts

* Measure model's performance
* Guide the learning process
* Provide optimization objective

#### Common Loss Functions

1. **Mean Squared Error (MSE)**
```
MSE = (1/n) ∑(yᵢ - ŷᵢ)²
```
* Used for regression problems
* Heavily penalizes large errors
* Sensitive to outliers

2. **Cross-Entropy**
```
CE = -∑(yᵢ log(ŷᵢ))
```
* Used for classification
* Measures probability distribution differences
* Better for binary/categorical outcomes

3. **Hinge Loss**
```
L = max(0, 1 - y·f(x))
```
* Used in Support Vector Machines
* Margin-based loss function
* Good for binary classification

### Likelihood Models

#### Key Concepts

* Based on probability theory
* Estimate data probability given parameters
* Allow for uncertainty quantification

#### Maximum Likelihood Estimation (MLE)

* Finds parameters maximizing data likelihood
* Often uses log-likelihood for computation
* Forms basis for many statistical methods

### Resources

* [Loss Functions Interactive Demo](https://www.desmos.com/calculator/1chunlr7dx)
* [Maximum Likelihood Tutorial](https://www.youtube.com/watch?v=XepXtl9YKwc)
* [Likelihood Estimation Guide](https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-and-maximum-a-posteriori-estimation-d7c318f9d22d)

### Regularization

Techniques to prevent overfitting by adding constraints or penalties to model parameters.

### Types of Regularization

#### L1 Regularization (Lasso)
```
Loss_L1 = Loss + λ∑|βᵢ|
```
* Adds absolute value of coefficients
* Can produce sparse models
* Feature selection capability

#### L2 Regularization (Ridge)
```
Loss_L2 = Loss + λ∑βᵢ²
```
* Adds squared magnitude of coefficients
* Shrinks coefficients toward zero
* Handles correlated features well

#### Elastic Net
```
Loss_Elastic = Loss + λ₁∑|βᵢ| + λ₂∑βᵢ²
```
* Combines L1 and L2 regularization
* Benefits of both approaches
* Two hyperparameters to tune

### Effects and Applications

#### Impact on Model
* Reduces model complexity
* Improves generalization
* Can prevent feature collinearity

#### Choosing Regularization Strength (λ)
* Too high: Underfitting
* Too low: Overfitting
* Requires cross-validation

### Resources

* [Regularization Effects Visualization](https://www.desmos.com/calculator/1chunlr7dx)
* [StatQuest: Ridge Regression](https://www.youtube.com/watch?v=Q81RR3yKn30)
* [Machine Learning Regularization Guide](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

### Parameter Optimization

Methods for finding optimal model parameters to minimize the loss function.

### Key Techniques

#### 1. Gradient Descent
* Updates parameters in direction of steepest decrease
* Learning rate determines step size
* Can get stuck in local minima

#### 2. Stochastic Gradient Descent (SGD)
* Updates using single random sample
* Faster and can escape local minima
* More noise in parameter updates

#### 3. Mini-batch Gradient Descent
* Updates using small random subset
* Balances speed and stability
* Most commonly used approach

#### 4. Newton's Method
* Uses second-order derivatives
* Faster convergence near minimum
* Computationally expensive

### Mathematical Formulation

For parameter θ:
```
θₜ₊₁ = θₜ - η∇L(θₜ)

Where:
- θₜ is current parameter value
- η is learning rate
- ∇L is gradient of loss function
```

### Optimization Challenges

* Local minima and saddle points
* Choosing appropriate learning rates
* Handling high-dimensional parameters
* Convergence criteria

### Resources

* [Gradient Descent Visualization](https://www.benfrederickson.com/numerical-optimization/)
* [SGD Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)
* [Optimization Algorithms Overview](https://ruder.io/optimizing-gradient-descent/)

### Large Dataset Optimization

Techniques for efficient training with large-scale datasets.

### Key Techniques

#### 1. Mini-batch Gradient Descent
* Uses small random data subsets
* Balances computation and update frequency
* Typically 32-512 samples per batch

#### 2. SGD with Momentum
```
v(t) = γv(t-1) + η∇L(θ)
θ(t) = θ(t-1) - v(t)

Where:
- γ is momentum coefficient
- η is learning rate
- ∇L is gradient of loss function
```
* Accelerates SGD in relevant directions
* Helps overcome local minima
* Reduces oscillation

#### 3. Adaptive Learning Methods

##### AdaGrad
* Adapts learning rate per parameter
* Accumulates squared gradients
* Good for sparse data

##### RMSprop
* Uses moving average of squared gradients
* Prevents learning rate decay
* Works well with non-stationary objectives

##### Adam
* Combines momentum and adaptive rates
* Includes bias correction
* Currently most popular optimizer

### Distributed Training

#### Data Parallelism
* Splits data across devices
* Each device computes gradients
* Synchronizes parameter updates

#### Model Parallelism
* Splits model across devices
* Reduces memory per device
* Requires careful pipeline design

### Resources

* [Optimization Algorithms Visualization](https://distill.pub/2017/momentum/)
* [Mini-batch Gradient Descent Tutorial](https://www.youtube.com/watch?v=4qJaSmvhxi8)
* [Large Scale ML Guide](https://towardsdatascience.com/large-scale-machine-learning-f54794f8d48f)

### Hyperparameter Optimization

Methods for finding optimal model hyperparameters.

### Common Hyperparameters

* Learning rate
* Number of hidden layers/neurons
* Regularization strength
* Kernel parameters
* Batch size

### Key Techniques

#### 1. Grid Search
* Exhaustive search through parameter space
* Tests all combinations of values
* Computationally expensive but thorough

#### 2. Random Search
```python
for i in range(num_trials):
    params = {
        'learning_rate': 10**uniform(-4, -2),
        'num_layers': randint(1, 5),
        'hidden_units': randint(32, 512)
    }
    score = train_and_evaluate(params)
```
* Randomly samples parameter space
* Often more efficient than grid search
* Better for high-dimensional spaces

#### 3. Bayesian Optimization
* Uses probabilistic model to guide search
* Balances exploration and exploitation
* More efficient for expensive evaluations

#### 4. Genetic Algorithms
* Evolves parameter combinations
* Uses natural selection principles
* Good for complex parameter spaces

### Best Practices

* Start with broad search space
* Use log-scale for numerical parameters
* Consider parameter dependencies
* Monitor computational resources

### Resources

* [Bayesian Optimization Visualization](https://distill.pub/2020/bayesian-optimization/)
* [Hyperparameter Tuning Tutorial](https://www.youtube.com/watch?v=4MK_OJJ82YI)
* [Comprehensive Optimization Guide](https://towardsdatascience.com/a-comprehensive-guide-to-hyperparameter-optimization-1b18175ebe78)

## Neural Networks & Deep Learning

### Neural Network Model

Computational models inspired by biological neural networks.

### Key Components

#### 1. Network Architecture

* **Input Layer**: Receives initial data
* **Hidden Layers**: Process information
* **Output Layer**: Produces final prediction
* **Neurons**: Basic computational units

#### 2. Activation Functions

##### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
* Most commonly used
* Helps solve vanishing gradient
* Computationally efficient

##### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
```
* Outputs between 0 and 1
* Used in binary classification
* Historical importance

##### Tanh
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
* Outputs between -1 and 1
* Zero-centered
* Stronger gradients than sigmoid

### Mathematical Representation

For a single neuron:
```
y = f(∑(wᵢxᵢ + b))

Where:
- y is output
- f is activation function
- wᵢ are weights
- xᵢ are inputs
- b is bias
```

### Advantages

* Can learn complex patterns
* Automatic feature extraction
* Universal function approximation
* Scalable to large datasets

### Disadvantages

* Requires large training data
* Computationally intensive
* Limited interpretability
* Many hyperparameters

### Resources

* [Neural Network Playground](https://playground.tensorflow.org/)
* [3Blue1Brown NN Series](https://www.youtube.com/watch?v=aircAruvnKk)
* [Beginner's Guide to Neural Networks](https://towardsdatascience.com/a-beginners-guide-to-neural-networks-d5cf7e369a13)

### Network Training

Training deep neural networks through backpropagation and optimization techniques.

### Key Concepts

#### 1. Backpropagation
* Efficiently computes gradients
* Uses chain rule for error propagation
* Core algorithm for neural network training

#### Mathematical Formulation
```
∂L/∂wⁱ = ∂L/∂aⁱ * ∂aⁱ/∂zⁱ * ∂zⁱ/∂wⁱ

Where:
- L is loss function
- w are weights
- a is activation
- z is weighted sum
```

#### 2. Optimization Algorithms

##### Gradient Descent Variants
* **Batch**: Uses entire dataset
* **Mini-batch**: Uses data subsets
* **Stochastic**: Uses single samples

##### Advanced Optimizers
* **Adam**: Adaptive moment estimation
* **RMSprop**: Root mean square propagation
* **AdaGrad**: Adaptive gradient algorithm

### Training Challenges

#### 1. Vanishing/Exploding Gradients
* Gradients become very small/large
* Deeper layers learn slowly/unstably
* Solutions: 
  - Proper initialization
  - Batch normalization
  - Residual connections

#### 2. Local Minima and Saddle Points
* Can trap optimization process
* More common in high dimensions
* Solutions:
  - Momentum-based optimizers
  - Learning rate scheduling
  - Proper initialization

### Resources

* [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)
* [Optimization Algorithms Overview](https://ruder.io/optimizing-gradient-descent/)
* [Training Visualization](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)

### Convolutional Neural Networks (CNNs)

Specialized neural networks for processing grid-like data, particularly images.

### Architecture Components

#### 1. Convolutional Layers
```
Output = ∑(Input * Kernel) + bias

Where:
- * denotes convolution operation
- Kernel is learnable filter
```
* Detect local patterns
* Parameter sharing
* Translation invariance

#### 2. Pooling Layers
* Reduce spatial dimensions
* Common types:
  - Max pooling
  - Average pooling
* Provides spatial invariance

#### 3. Fully Connected Layers
* Traditional neural network layers
* Usually at network end
* Combine features for final prediction

### Popular Architectures

#### 1. LeNet-5
* Pioneer CNN architecture
* 7 layers (including input)
* Handwritten digit recognition

#### 2. AlexNet
* Breakthrough in 2012 ImageNet
* ReLU activation
* Dropout regularization

#### 3. VGGNet
* Simple, uniform architecture
* 3x3 convolutions throughout
* Very deep (16-19 layers)

#### 4. ResNet
* Introduced residual connections
* Very deep (50-152 layers)
* Solved vanishing gradient problem

### Applications

* Image Classification
* Object Detection
* Facial Recognition
* Medical Image Analysis
* Video Processing

### Resources

* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
* [CNN Architecture Guide](https://www.youtube.com/watch?v=YRhxdVk_sIs)
* [Comprehensive CNN Guide](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

### Dropout and Regularization

Techniques to prevent overfitting in neural networks.

### Dropout

#### Key Concepts
* Randomly "drops" neurons during training
* Prevents co-adaptation of neurons
* Acts as model averaging technique

#### Implementation
```python
p = dropout_rate  # typically 0.2-0.5
mask = np.random.binomial(1, p, size=layer_size)
output = input * mask * (1/p)  # Scale to maintain expected value
```

#### Training Phase
* Neurons randomly deactivated
* Remaining neurons scaled up
* Different dropout masks each batch

#### Testing Phase
* All neurons active
* Outputs scaled by dropout probability
* Approximates model averaging

### Other Regularization Techniques

#### 1. L1/L2 Regularization
```
Loss = Original_Loss + λ₁∑|w| + λ₂∑w²
```
* Penalizes large weights
* Encourages simpler models
* Helps prevent overfitting

#### 2. Batch Normalization
```
y = γ((x - μ)/σ) + β

Where:
- μ is batch mean
- σ is batch standard deviation
- γ, β are learnable parameters
```
* Normalizes layer inputs
* Reduces internal covariate shift
* Speeds up training

#### 3. Early Stopping
* Monitor validation performance
* Stop when performance plateaus
* Prevent overfitting automatically

### Resources

* [Dropout Animation](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
* [Dropout Explained](https://www.youtube.com/watch?v=ARq74QuavAo)
* [Regularization Guide](https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275)

### Advanced Neural Network Architectures

Modern neural network architectures designed for specific tasks and improved performance.

### Recurrent Neural Networks (RNNs)

Networks designed to work with sequential data by maintaining internal state.

#### Basic RNN Architecture
```
hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b)
yₜ = Wᵧhₜ + bᵧ

Where:
- hₜ is hidden state at time t
- xₜ is input at time t
- yₜ is output at time t
- W, b are weights and biases
```

#### LSTM (Long Short-Term Memory)

##### Architecture Components
* **Forget Gate**: Controls information to discard
* **Input Gate**: Controls new information to store
* **Output Gate**: Controls output information
* **Memory Cell**: Maintains long-term dependencies

##### Mathematical Formulation
```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
cₜ = fₜ * cₜ₋₁ + iₜ * c̃ₜ
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
hₜ = oₜ * tanh(cₜ)
```

#### GRU (Gated Recurrent Unit)
* Simplified version of LSTM
* Combines forget and input gates
* Merges cell and hidden states

### Resources
* [RNN Visualization](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1/)
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [GRU vs LSTM](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

### Transformer Architecture

State-of-the-art architecture for sequence processing based on attention mechanisms.

#### Key Components

##### 1. Self-Attention
```
Attention(Q, K, V) = softmax(QK^T/√dk)V

Where:
- Q: Query matrix
- K: Key matrix
- V: Value matrix
- dk: Dimension of keys
```

##### 2. Multi-Head Attention
* Parallel attention computations
* Different representation subspaces
* Improved modeling capability

##### 3. Position-wise Feed-Forward
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
* Applied to each position separately
* Consists of two linear transformations

#### Architecture Features

##### 1. Encoder
* Multiple identical layers
* Each layer has:
  - Multi-head attention
  - Feed-forward network
  - Layer normalization
  - Residual connections

##### 2. Decoder
* Similar to encoder but includes:
  - Masked multi-head attention
  - Cross-attention with encoder
  - Additional layer normalization

### Applications
* Machine Translation
* Text Generation
* Document Understanding
* Code Generation

### Resources
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Transformer Visualization](http://jalammar.github.io/illustrated-transformer/)
* [Transformer Implementation Guide](https://www.youtube.com/watch?v=4Bdc55j80l8)

### Transfer Learning and Pre-trained Models

Using knowledge from pre-trained models for new tasks.

### Key Concepts

#### 1. Pre-training
* Train model on large dataset
* Learn general features
* Often uses self-supervised learning

#### 2. Fine-tuning
* Adapt pre-trained model to new task
* Update some or all parameters
* Requires less data than training from scratch

### Common Approaches

#### 1. Feature Extraction
* Freeze pre-trained layers
* Only train new classification layers
* Faster and prevents overfitting

#### 2. Fine-tuning
* Unfreeze some pre-trained layers
* Update weights for new task
* Requires careful learning rate selection

#### 3. Progressive Fine-tuning
* Gradually unfreeze layers
* Start from top layers
* Better preservation of learned features

### Popular Pre-trained Models

#### Computer Vision
* ResNet
* VGG
* EfficientNet
* YOLO

#### Natural Language Processing
* BERT
* GPT
* RoBERTa
* T5

### Best Practices

* Start with frozen pre-trained weights
* Use small learning rate for fine-tuning
* Monitor validation performance
* Consider domain similarity
* Use appropriate augmentation

### Resources
* [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)
* [Fine-tuning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
* [Pre-trained Models Overview](https://towardsdatascience.com/a-comprehensive-guide-to-transfer-learning-4466456e9ed)

## Ensemble Methods

Techniques that combine multiple models to create more robust and accurate predictions.

### Bagging (Bootstrap Aggregating)

Method that trains models on random subsets of the training data.

#### Key Concepts

* **Bootstrap Sampling**: Random sampling with replacement
* **Parallel Training**: Models trained independently
* **Aggregation**: Combine predictions through voting or averaging
* **Variance Reduction**: Reduces model variance through averaging

#### Mathematical Formulation

For regression with m models:
```
ŷ(x) = (1/m) ∑ᵢ fᵢ(x)

Where:
- ŷ is final prediction
- fᵢ is i-th model prediction
- m is number of models
```

#### Advantages
* Reduces overfitting
* Parallel training possible
* Handles high variance well

#### Disadvantages
* Computationally intensive
* Requires more memory
* May lose interpretability

### Resources
* [Bagging Visualization](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html)
* [Bootstrap Aggregating Tutorial](https://www.youtube.com/watch?v=2Mg8QD0F1dQ)
* [Ensemble Methods Guide](https://towardsdatascience.com/ensemble-learning-bagging-boosting-3098079e5422)

### Random Forests

Ensemble method combining decision trees with random feature selection.

#### Algorithm Steps

1. Create bootstrap sample of training data
2. For each node in tree:
   * Select random subset of features
   * Find best split among these features
   * Split node
3. Repeat until tree is fully grown
4. Aggregate predictions from all trees

#### Mathematical Details
```python
# Feature selection at each node
n_features_subset = int(sqrt(total_features))  # typical for classification
# or
n_features_subset = int(total_features/3)      # typical for regression

# Final prediction (classification)
prediction = mode([tree.predict(x) for tree in forest])
# or (regression)
prediction = mean([tree.predict(x) for tree in forest])
```

#### Hyperparameters
* Number of trees
* Maximum depth
* Minimum samples per leaf
* Number of features per split

### Resources
* [Random Forest Visualization](https://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
* [RF Algorithm Explained](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
* [Understanding Random Forests](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)

### Boosting Methods

Sequential ensemble methods that focus on misclassified instances.

#### AdaBoost (Adaptive Boosting)

##### Algorithm
1. Initialize equal sample weights
2. For each iteration:
   * Train weak learner on weighted data
   * Calculate error and model weight
   * Update sample weights
3. Combine models using weights

##### Mathematical Formulation
```
Final Prediction = sign(∑ᵢ αᵢhᵢ(x))

Where:
- αᵢ is weight of i-th model
- hᵢ(x) is prediction of i-th model
- Model weight: αᵢ = 0.5 * ln((1-εᵢ)/εᵢ)
- εᵢ is weighted error rate
```

#### Gradient Boosting

##### Key Components
* Sequential addition of weak learners
* Each learner fits residuals of previous models
* Gradient descent in function space

##### Mathematical Details
```
F(x) = F₀(x) + ∑ᵢ γᵢhᵢ(x)

Where:
- F₀ is initial prediction
- hᵢ are weak learners
- γᵢ are step sizes
```

#### XGBoost Features
* Regularization terms
* Tree pruning
* Parallel processing
* Cache awareness
* Sparsity awareness

### Implementation Examples

#### Gradient Boosting Implementation
```python
class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        
    def fit(self, X, y):
        # Initialize prediction
        F0 = np.mean(y)
        current_prediction = F0
        
        for _ in range(self.n_estimators):
            # Calculate negative gradient
            residuals = y - current_prediction
            # Fit base learner
            model = DecisionTreeRegressor()
            model.fit(X, residuals)
            self.models.append(model)
            # Update predictions
            current_prediction += self.learning_rate * model.predict(X)
```

### Best Practices

#### Model Selection
* Random Forests: Good default choice
* XGBoost: When performance is critical
* AdaBoost: For weak base learners

#### Hyperparameter Tuning
* Number of estimators
* Learning rate (for boosting)
* Maximum depth of trees
* Minimum samples per leaf

#### Cross-validation Strategies
* K-fold for smaller datasets
* Out-of-bag for random forests
* Hold-out for large datasets

### Resources
* [Gradient Boosting Interactive](https://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [Ensemble Learning Guide](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

# Non-linear Transformations & Kernels

## Non-linear Input Transformations

Methods to capture complex relationships by transforming input features into higher-dimensional spaces.

### Key Concepts

#### Feature Engineering
* Creating new features through transformations
* Capturing non-linear relationships
* Expanding feature space

#### Common Transformations

1. **Polynomial Features**
```python
def polynomial_transform(x, degree=2):
    # Example for 2D input [x₁, x₂]
    if degree == 2:
        return [x₁, x₂, x₁², x₁x₂, x₂²]
```

2. **Logarithmic**
```
f(x) = log(x + c)  # c prevents undefined log(0)
```

3. **Exponential**
```
f(x) = exp(x)
```

4. **Trigonometric**
```
f(x) = [sin(x), cos(x), tan(x)]
```

### Advantages
* Allows linear models to capture non-linear patterns
* Interpretable transformations
* Can improve model performance

### Disadvantages
* Curse of dimensionality
* Risk of overfitting
* Feature selection becomes crucial

### Resources
* [Feature Engineering Guide](https://www.youtube.com/watch?v=6WDFfaYtN6s)
* [Non-linear Transformations Tutorial](https://towardsdatascience.com/feature-engineering-a-comprehensive-overview-a7ad04d2f25e)

## Kernel Methods

Techniques to implicitly compute transformations in high-dimensional spaces.

### Kernel Functions

#### Common Kernels

1. **Linear Kernel**
```
K(x, y) = x^T y
```

2. **Polynomial Kernel**
```
K(x, y) = (γx^T y + c)^d

Parameters:
- γ: Scale parameter
- c: Bias term
- d: Polynomial degree
```

3. **RBF (Gaussian) Kernel**
```
K(x, y) = exp(-γ||x - y||²)

Where:
- γ: Determines influence radius
```

4. **Sigmoid Kernel**
```
K(x, y) = tanh(γx^T y + c)
```

### Kernel Ridge Regression

Combines ridge regression with kernel trick.

#### Mathematical Formulation
```
f(x) = ∑ᵢ αᵢK(x, xᵢ)

Where:
- αᵢ are dual coefficients
- K is kernel function
```

#### Implementation
```python
class KernelRidge:
    def __init__(self, kernel='rbf', gamma=1.0, alpha=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha
    
    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        # Solve for dual coefficients
        self.dual_coef_ = np.linalg.solve(
            K + self.alpha * np.eye(n_samples), 
            y
        )
    
    def predict(self, X):
        K = self._compute_kernel(X, self.X_train)
        return K.dot(self.dual_coef_)
```

### Support Vector Regression (SVR)

Extension of SVM for regression tasks.

#### Objective Function
```
minimize: (1/2)||w||² + C∑(ξᵢ + ξᵢ*)

subject to:
|y - (w^T x + b)| ≤ ε + ξ
ξᵢ, ξᵢ* ≥ 0
```

#### Key Parameters
* C: Regularization strength
* ε: Margin of tolerance
* Kernel parameters

### Practical Considerations

#### Kernel Selection
* Linear: For linearly separable data
* RBF: Good default choice
* Polynomial: When feature interactions matter
* Custom: Domain-specific knowledge

#### Parameter Tuning
* Grid search with cross-validation
* Bayesian optimization
* Random search for large parameter spaces

#### Scaling
* Always scale input features
* Consider standardization or normalization
* Maintain consistent scaling for test data

### Resources
* [Kernel Methods Math](https://www.youtube.com/watch?v=XUj5JbQihlU)
* [SVR Tutorial](https://www.youtube.com/watch?v=Y6RRHw9uN9o)
* [Kernel Selection Guide](https://towardsdatascience.com/understanding-kernel-trick-5e18d0adf897)

## Kernel Theory

Mathematical foundations of kernel methods.

### Key Concepts

#### 1. Mercer's Theorem
* Conditions for valid kernel functions
* Relationship to feature spaces
* Positive semi-definiteness

#### 2. Reproducing Kernel Hilbert Spaces (RKHS)
* Function space associated with kernel
* Properties and operations
* Theoretical foundation

### Mathematical Foundations
```
K(x, y) = <φ(x), φ(y)>

Where:
- φ is feature map
- <·,·> is inner product
- K is kernel function
```

### Resources
* [RKHS Tutorial](https://www.youtube.com/watch?v=_PwhiWxHK8o)
* [Kernel Theory Guide](https://towardsdatascience.com/understanding-kernel-trick-5e18d0adf897)

# Bayesian Methods

## Bayesian Approach

A probabilistic approach to machine learning that incorporates prior knowledge and uncertainty.

### Key Concepts

#### 1. Prior Probability
* Initial beliefs before observing data
* Encodes domain knowledge
* Can be informative or non-informative

#### 2. Likelihood
* Probability of data given parameters
* Model assumptions
* Measurement process

#### 3. Posterior Probability
* Updated beliefs after observing data
* Combines prior and likelihood
* Basis for predictions

### Bayes' Theorem

```
P(θ|D) = P(D|θ)P(θ)/P(D)

Where:
- P(θ|D): Posterior probability
- P(D|θ): Likelihood
- P(θ): Prior probability
- P(D): Evidence (normalizing constant)
```

### Implementation Example
```python
class BayesianEstimator:
    def __init__(self, prior):
        self.prior = prior
    
    def update(self, data):
        likelihood = self.compute_likelihood(data)
        self.posterior = likelihood * self.prior
        self.posterior /= self.posterior.sum()  # Normalize
        self.prior = self.posterior  # Update for next iteration
```

### Resources
* [Bayesian Statistics Visualization](https://seeing-theory.brown.edu/bayesian-inference/index.html)
* [StatQuest: Bayes Theorem](https://www.youtube.com/watch?v=0F0QoMCSKJ4)
* [Bayesian Methods Guide](https://machinelearningmastery.com/introduction-to-bayesian-belief-networks/)

## Bayesian Linear Regression

Probabilistic approach to linear regression providing uncertainty estimates.

### Model Formulation

#### Likelihood
```
y|X,w,σ² ~ N(Xw, σ²I)
```

#### Prior
```
w ~ N(μ₀, Σ₀)
```

#### Posterior
```
w|X,y,σ² ~ N(μₙ, Σₙ)

Where:
μₙ = Σₙ(Σ₀⁻¹μ₀ + σ⁻²X^Ty)
Σₙ = (Σ₀⁻¹ + σ⁻²X^TX)⁻¹
```

### Implementation
```python
class BayesianLinearRegression:
    def __init__(self, prior_mean, prior_cov, noise_var):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.noise_var = noise_var
        
    def fit(self, X, y):
        # Compute posterior parameters
        self.posterior_cov = np.linalg.inv(
            np.linalg.inv(self.prior_cov) + 
            X.T @ X / self.noise_var
        )
        self.posterior_mean = self.posterior_cov @ (
            np.linalg.inv(self.prior_cov) @ self.prior_mean +
            X.T @ y / self.noise_var
        )
    
    def predict(self, X, return_std=False):
        mean = X @ self.posterior_mean
        if return_std:
            std = np.sqrt(np.diag(
                X @ self.posterior_cov @ X.T + 
                self.noise_var * np.eye(X.shape[0])
            ))
            return mean, std
        return mean
```

### Advantages
* Uncertainty quantification
* Natural regularization
* Handles missing data well
* Incorporates prior knowledge

### Disadvantages
* Computationally intensive
* Prior specification needed
* Complex for large datasets

### Resources
* [Bayesian Linear Regression Tutorial](https://www.youtube.com/watch?v=nrd4AnDLR3U)
* [Probabilistic ML Guide](https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-performance-ac0567b9c763)

## Gaussian Processes

Non-parametric approach defining distributions over functions.

### Key Components

#### 1. Mean Function
```
m(x) = E[f(x)]
```

#### 2. Covariance Function (Kernel)
```
k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]
```

### Mathematical Formulation
```
f(x) ~ GP(m(x), k(x,x'))

Posterior:
f*|X,y,X* ~ N(μ*, Σ*)

Where:
μ* = m(X*) + K*K⁻¹(y - m(X))
Σ* = K** - K*K⁻¹K*^T
```

### Common Kernels

#### RBF (Squared Exponential)
```
k(x,x') = σ²exp(-||x-x'||²/(2l²))
```

#### Matérn Kernel
```
k(x,x') = σ²(1 + √3r/l)exp(-√3r/l)
Where r = ||x-x'||
```

### Implementation
```python
class GaussianProcess:
    def __init__(self, kernel, noise=1e-5):
        self.kernel = kernel
        self.noise = noise
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(
            self.L.T, 
            np.linalg.solve(self.L, y)
        )
    
    def predict(self, X_new, return_std=False):
        K_new = self.kernel(X_new, self.X_train)
        mean = K_new @ self.alpha
        
        if return_std:
            v = np.linalg.solve(self.L, K_new.T)
            var = self.kernel(X_new, X_new) - v.T @ v
            return mean, np.sqrt(np.diag(var))
        return mean
```

### Applications
* Regression with uncertainty
* Bayesian optimization
* Active learning
* Time series forecasting

### Resources
* [GP Visualization](http://www.gaussianprocess.org/gpml/chapters/RW2.html)
* [Deep GP Tutorial](https://www.youtube.com/watch?v=4vGiHC35j9s)
* [Visual GP Guide](https://distill.pub/2019/visual-exploration-gaussian-processes/)

# Generative & Unsupervised Learning

## Gaussian Mixture Models & Discriminant Analysis

### Gaussian Mixture Models (GMMs)

Probabilistic models representing data as a mixture of Gaussian distributions.

#### Mathematical Foundation

##### Mixture Model
```
p(x) = ∑ᵢ πᵢN(x|μᵢ,Σᵢ)

Where:
- πᵢ: Mixing coefficients (∑πᵢ = 1)
- μᵢ: Mean vectors
- Σᵢ: Covariance matrices
- N(): Gaussian distribution
```

#### EM Algorithm Steps

1. **Initialization**
```python
def initialize_gmm(X, n_components):
    n_samples, n_features = X.shape
    # Initialize means using k-means++
    means = kmeans_plus_plus(X, n_components)
    # Initialize covariances as identity matrices
    covs = [np.eye(n_features) for _ in range(n_components)]
    # Initialize mixing coefficients uniformly
    weights = np.ones(n_components) / n_components
    return means, covs, weights
```

2. **E-Step**: Compute responsibilities
```python
def e_step(X, means, covs, weights):
    responsibilities = np.zeros((X.shape[0], len(weights)))
    for k in range(len(weights)):
        responsibilities[:,k] = weights[k] * multivariate_normal.pdf(
            X, means[k], covs[k]
        )
    # Normalize
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities
```

3. **M-Step**: Update parameters
```python
def m_step(X, responsibilities):
    n_samples, n_features = X.shape
    n_components = responsibilities.shape[1]
    
    Nk = responsibilities.sum(axis=0)
    means = np.dot(responsibilities.T, X) / Nk[:,np.newaxis]
    
    covs = []
    for k in range(n_components):
        diff = X - means[k]
        covs.append(np.dot(responsibilities[:,k] * diff.T, diff) / Nk[k])
    
    weights = Nk / n_samples
    return means, covs, weights
```

#### Applications
* Clustering
* Density Estimation
* Anomaly Detection
* Image Segmentation

#### Resources
* [GMM Animation](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html)
* [EM Algorithm Tutorial](https://www.youtube.com/watch?v=qnDeCzwH2_E)
* [GMM Implementation Guide](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95)

### Discriminant Analysis

#### Linear Discriminant Analysis (LDA)

##### Key Concepts
* Assumes equal covariance matrices
* Maximizes class separability
* Reduces dimensionality

##### Mathematical Formulation
```
Between-class scatter: Sᵦ = ∑ᵢ Nᵢ(μᵢ-μ)(μᵢ-μ)ᵀ
Within-class scatter: Sᵥ = ∑ᵢ ∑ₓ∈Cᵢ (x-μᵢ)(x-μᵢ)ᵀ
Objective: maximize tr(Sᵥ⁻¹Sᵦ)
```

##### Implementation
```python
class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        
    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        # Compute means
        mean_overall = np.mean(X, axis=0)
        mean_classes = [np.mean(X[y==c], axis=0) for c in class_labels]
        
        # Compute scatter matrices
        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))
        
        for idx, c in enumerate(class_labels):
            X_c = X[y==c]
            mean_c = mean_classes[idx]
            
            # Within class scatter
            S_w += np.dot((X_c - mean_c).T, (X_c - mean_c))
            
            # Between class scatter
            n_c = len(X_c)
            mean_diff = (mean_c - mean_overall).reshape(-1,1)
            S_b += n_c * np.dot(mean_diff, mean_diff.T)
        
        # Solve eigenvalue problem
        eig_vals, eig_vecs = np.linalg.eigh(
            np.linalg.inv(S_w).dot(S_b)
        )
        
        # Sort eigenvectors
        idx = np.argsort(eig_vals)[::-1]
        self.components = eig_vecs[:,idx[:self.n_components]]
```

#### Quadratic Discriminant Analysis (QDA)

##### Key Differences from LDA
* Allows different covariance matrices per class
* More flexible decision boundaries
* Requires more parameters

##### Decision Rule
```
Assign x to class k if:
-0.5 log|Σₖ| - 0.5(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) + log(πₖ) is maximum
```

#### Resources
* [LDA vs QDA Visualization](https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html)
* [Discriminant Analysis Tutorial](https://www.youtube.com/watch?v=azXCzI57Yfc)
* [LDA Implementation Guide](https://towardsdatascience.com/linear-discriminant-analysis-explained-f88be6c1e00b)

## Cluster Analysis

### K-means Clustering

Partitioning method that groups data into k clusters based on distance to centroids.

#### Algorithm

```python
class KMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
    
    def fit(self, X):
        # Initialize centroids using k-means++
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
    
    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        new_centroids = np.array([
            X[labels == k].mean(axis=0)
            for k in range(self.n_clusters)
        ])
        return new_centroids
```

#### Complexity
* Time: O(kndi)
  - k: number of clusters
  - n: number of points
  - d: dimensions
  - i: iterations

#### Advantages
* Simple to understand
* Fast for small datasets
* Guaranteed convergence

#### Disadvantages
* Needs predefined k
* Sensitive to outliers
* Assumes spherical clusters

### Hierarchical Clustering

Builds a tree of clusters, either bottom-up (agglomerative) or top-down (divisive).

#### Agglomerative Clustering

```python
class AgglomerativeClustering:
    def __init__(self, n_clusters, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def fit(self, X):
        # Initialize each point as a cluster
        n_samples = X.shape[0]
        clusters = [[i] for i in range(n_samples)]
        distances = self._compute_distances(X)
        
        while len(clusters) > self.n_clusters:
            # Find closest clusters
            i, j = self._find_closest_clusters(distances)
            
            # Merge clusters
            clusters[i].extend(clusters[j])
            clusters.pop(j)
            
            # Update distances
            distances = self._update_distances(distances, i, j, X, clusters)
            
        return clusters
    
    def _compute_distances(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distances[i,j] = distances[j,i] = np.linalg.norm(X[i] - X[j])
        return distances
```

#### Linkage Methods
* Single: Minimum distance between clusters
* Complete: Maximum distance between clusters
* Average: Mean distance between clusters
* Ward: Minimizes variance within clusters

### DBSCAN (Density-Based Spatial Clustering)

Clusters points based on density, automatically determining number of clusters.

#### Algorithm Implementation

```python
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(self, X):
        self.labels_ = np.full(len(X), -1)  # -1 indicates unvisited
        cluster_id = 0
        
        for point_idx in range(len(X)):
            if self.labels_[point_idx] != -1:
                continue
                
            neighbors = self._find_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = 0  # Noise point
                continue
                
            # Expand cluster
            cluster_id += 1
            self._expand_cluster(X, point_idx, neighbors, cluster_id)
        
        return self.labels_
    
    def _find_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels_[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            current_point = neighbors[i]
            
            if self.labels_[current_point] == 0:
                self.labels_[current_point] = cluster_id
            
            elif self.labels_[current_point] == -1:
                self.labels_[current_point] = cluster_id
                new_neighbors = self._find_neighbors(X, current_point)
                
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])
            
            i += 1
```

### Advanced Clustering Methods

#### Spectral Clustering

Uses eigenvalues of similarity matrix to reduce dimensionality before clustering.

```python
class SpectralClustering:
    def __init__(self, n_clusters, affinity='rbf'):
        self.n_clusters = n_clusters
        self.affinity = affinity
    
    def fit(self, X):
        # Compute affinity matrix
        A = self._get_affinity_matrix(X)
        
        # Compute Laplacian
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        
        # Compute eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(L)
        
        # Use k smallest eigenvectors
        features = eigenvecs[:, :self.n_clusters]
        
        # Apply k-means
        kmeans = KMeans(n_clusters=self.n_clusters)
        return kmeans.fit_predict(features)
```

#### Mean Shift

Finds clusters by identifying modes of density function.

```python
def mean_shift(X, bandwidth):
    centers = []
    for point in X:
        center = point
        while True:
            points_within = X[np.linalg.norm(X - center, axis=1) < bandwidth]
            new_center = points_within.mean(axis=0)
            
            if np.allclose(new_center, center):
                break
            center = new_center
        centers.append(center)
    return np.array(centers)
```

### Resources
* [Interactive Clustering Demo](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
* [Clustering Comparison](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
* [DBSCAN Explained](https://www.youtube.com/watch?v=RDZUdRSDOok)

## Deep Generative Models

### Variational Autoencoders (VAEs)

Neural network architecture that learns to generate data by modeling its probability distribution.

#### Architecture Components

1. **Encoder Network**
```python
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # Output mean and log variance
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
```

2. **Decoder Network**
```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.decoder(z)
```

#### Loss Function
```python
def vae_loss(x_recon, x, mu, log_var):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss
```

#### Training Process
```python
def train_vae(vae, optimizer, dataloader):
    vae.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, mu, log_var = vae(batch)
        
        # Compute loss
        loss = vae_loss(x_recon, batch, mu, log_var)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
```

### Generative Adversarial Networks (GANs)

Framework where two networks compete: generator creates samples, discriminator evaluates them.

#### Basic Architecture

1. **Generator Network**
```python
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)
```

2. **Discriminator Network**
```python
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
```

#### Training Loop
```python
def train_gan(generator, discriminator, g_optimizer, d_optimizer, dataloader):
    criterion = nn.BCELoss()
    
    for real_data in dataloader:
        batch_size = real_data.size(0)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        label_real = torch.ones(batch_size, 1)
        label_fake = torch.zeros(batch_size, 1)
        
        # Real data
        output_real = discriminator(real_data)
        d_loss_real = criterion(output_real, label_real)
        
        # Fake data
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)
        output_fake = discriminator(fake_data.detach())
        d_loss_fake = criterion(output_fake, label_fake)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        
        output_fake = discriminator(fake_data)
        g_loss = criterion(output_fake, label_real)
        
        g_loss.backward()
        g_optimizer.step()
```

### Normalizing Flows

Sequence of invertible transformations that map simple distributions to complex ones.

#### Basic Components

1. **Flow Layer**
```python
class FlowLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Transform x
        y, log_det = self.transform(x)
        return y, log_det
    
    def inverse(self, y):
        # Inverse transform
        x = self.inverse_transform(y)
        return x
```

2. **Planar Flow**
```python
class PlanarFlow(FlowLayer):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))
        self.u = nn.Parameter(torch.randn(dim))
        
    def transform(self, x):
        activation = torch.tanh(torch.sum(self.w * x, dim=1) + self.b)
        y = x + self.u * activation.unsqueeze(1)
        
        # Compute log determinant
        phi = (1 - activation**2) * self.w
        log_det = torch.log(torch.abs(1 + torch.sum(self.u * phi, dim=1)))
        
        return y, log_det
```

### Diffusion Models

Generate data by gradually denoising random noise through a sequence of steps.

#### Implementation

1. **Forward Process (Noise Addition)**
```python
def forward_diffusion(x_0, t, beta_schedule):
    # Get noise schedule
    alpha_t = 1 - beta_schedule[t]
    sqrt_alpha_t = torch.sqrt(alpha_t)
    
    # Add noise
    epsilon = torch.randn_like(x_0)
    x_t = sqrt_alpha_t * x_0 + torch.sqrt(1 - alpha_t) * epsilon
    
    return x_t, epsilon
```

2. **Reverse Process (Denoising)**
```python
class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet(
            dim=64,
            channels=3,
            dim_mults=(1, 2, 4, 8)
        )
    
    def forward(self, x_t, t):
        return self.net(x_t, t)

def reverse_diffusion(model, x_t, t, beta_schedule):
    # Predict noise
    predicted_noise = model(x_t, t)
    
    # Remove noise
    alpha_t = 1 - beta_schedule[t]
    x_t_prev = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
    
    if t > 0:
        noise = torch.randn_like(x_t)
        x_t_prev = x_t_prev + beta_schedule[t-1] * noise
        
    return x_t_prev
```

### Resources
* [VAE Tutorial](https://arxiv.org/abs/1312.6114)
* [GAN Paper](https://arxiv.org/abs/1406.2661)
* [Diffusion Models](https://arxiv.org/abs/2006.11239)
* [Interactive GAN Lab](https://poloclub.github.io/ganlab/)
* [Normalizing Flows Tutorial](https://arxiv.org/abs/1908.09257)

# Representation Learning

## Autoencoders

Neural networks that learn compressed data representations through self-supervised learning.

### Basic Autoencoder Architecture

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        # Decode
        reconstruction = self.decoder(z)
        return reconstruction
```

### Denoising Autoencoder

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim)
        
    def add_noise(self, x, noise_factor=0.3):
        noise = torch.randn_like(x) * noise_factor
        corrupted_x = x + noise
        return torch.clamp(corrupted_x, 0., 1.)
    
    def forward(self, x):
        # Add noise
        corrupted_x = self.add_noise(x)
        # Reconstruct
        reconstruction = self.autoencoder(corrupted_x)
        return reconstruction, corrupted_x
```

### Sparse Autoencoder

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sparsity_param=0.05):
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim)
        self.sparsity_param = sparsity_param
        
    def kl_divergence(self, rho, rho_hat):
        return rho * torch.log(rho/rho_hat) + \
               (1-rho) * torch.log((1-rho)/(1-rho_hat))
    
    def forward(self, x):
        # Get latent representation
        z = self.autoencoder.encoder(x)
        # Calculate average activation
        rho_hat = torch.mean(z, dim=0)
        # Calculate sparsity penalty
        sparsity_penalty = self.kl_divergence(self.sparsity_param, rho_hat)
        # Get reconstruction
        reconstruction = self.autoencoder.decoder(z)
        return reconstruction, sparsity_penalty
```

## Self-Supervised Learning

Learning representations without explicit labels by using the data's inherent structure.

### Pretext Tasks

#### Rotation Prediction
```python
class RotationPredictor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, 4)  # 4 rotations
        
    def forward(self, x):
        features = self.backbone(x)
        rotation_pred = self.classifier(features)
        return rotation_pred

def create_rotations(x):
    rotations = [
        x,  # 0 degrees
        torch.rot90(x, k=1, dims=[2,3]),  # 90 degrees
        torch.rot90(x, k=2, dims=[2,3]),  # 180 degrees
        torch.rot90(x, k=3, dims=[2,3])   # 270 degrees
    ]
    return torch.cat(rotations, dim=0)
```

### Contrastive Learning

#### SimCLR Framework
```python
class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        self.backbone = backbone
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection(features)
        return F.normalize(projections, dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    # Concatenate representations
    z = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix
    sim = torch.mm(z, z.t().contiguous()) / temperature
    
    # Create labels for positive pairs
    batch_size = z1.size(0)
    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels + batch_size, labels])
    
    # Compute loss
    loss = F.cross_entropy(sim, labels)
    return loss
```

## Word Embeddings

### Word2Vec Implementation

#### Skip-gram Model
```python
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        embeds = self.embeddings(x)
        output = self.output(embeds)
        return output

def negative_sampling_loss(input_vectors, output_vectors, noise_vectors):
    batch_size, embed_size = input_vectors.shape
    
    # Positive samples
    pos_loss = torch.bmm(input_vectors.view(batch_size, 1, embed_size),
                        output_vectors.view(batch_size, embed_size, 1))
    pos_loss = F.logsigmoid(pos_loss).squeeze()
    
    # Negative samples
    neg_loss = torch.bmm(input_vectors.view(batch_size, 1, embed_size),
                        noise_vectors.transpose(1, 2))
    neg_loss = F.logsigmoid(-neg_loss).squeeze().sum(1)
    
    return -(pos_loss + neg_loss).mean()
```

### GloVe Implementation

```python
class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_biases = nn.Embedding(vocab_size, 1)
        self.c_biases = nn.Embedding(vocab_size, 1)
        
    def forward(self, i, j):
        w_i = self.w_embeddings(i)
        c_j = self.c_embeddings(j)
        w_bias = self.w_biases(i)
        c_bias = self.c_biases(j)
        
        return torch.sum(w_i * c_j, dim=1) + w_bias.squeeze() + c_bias.squeeze()

def glove_loss(predictions, log_cooccurrences, weights):
    loss = weights * (predictions - log_cooccurrences) ** 2
    return torch.mean(loss)
```

### FastText Extension

```python
class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ngram_vocab_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ngram_embeddings = nn.Embedding(ngram_vocab_size, embedding_dim)
        
    def forward(self, word_ids, ngram_ids):
        word_embeds = self.word_embeddings(word_ids)
        ngram_embeds = self.ngram_embeddings(ngram_ids)
        return word_embeds + torch.mean(ngram_embeds, dim=1)
```

### Best Practices

1. **Data Preprocessing**
   * Clean and normalize text
   * Handle rare words
   * Generate n-grams for FastText

2. **Training**
   * Use large corpus
   * Proper window size
   * Negative sampling ratio
   * Learning rate scheduling

3. **Evaluation**
   * Intrinsic (similarity, analogy tasks)
   * Extrinsic (downstream tasks)
   * Visualization (t-SNE, PCA)

### Resources
* [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
* [GloVe Project](https://nlp.stanford.edu/projects/glove/)
* [FastText Library](https://fasttext.cc/)
* [Embedding Projector](http://projector.tensorflow.org/)
* [SimCLR Paper](https://arxiv.org/abs/2002.05709)

