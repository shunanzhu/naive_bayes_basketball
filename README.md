# Naive Bayes Classifier Implementation Report

## To Run:
```bash
chmod +x NaiveBayesClassifier.sh

./NaiveBayesClassifier.sh Data/train_data.csv Data/validation_data.csv
```

## Architecture
The project centers around a custom Naive Bayes classifier implemented with Python 3.12.4. The core component is the CustomNaiveBayesClassifier class, which encapsulates all functionalities required for training and prediction. Key methods and their functionalities include:

### Data Handling Functions:

- ```preprocess_data``` Functionality: Reads and preprocesses training and validation data from CSV files.

- ```main``` Functionality: The main execution function that orchestrates data preprocessing, model training, prediction, and evaluation.


### Naive Bayes Class:

- ```__init__```: Initializes instance variables for class priors, mean and variance of numerical features, probabilities of categorical features, and indices for categorical and numerical features.
    
    Details:

    - ```self._unique_classes```: Stores the unique class labels from the training data.
    - ```self._class_priors```: Holds the prior probabilities of each class.
    - ```self._mean_values``` and self._variance_values: Arrays to store mean and variance of numerical features for each class.
    - ```self._categorical_probabilities```: List of dictionaries to store conditional probabilities for categorical features.
    - ```self._categorical_indices``` and ```self._numerical_indices```: Lists to keep track of categorical and numerical feature indices, respectively.

- ```set_weights```: Intializes the weights of the model; these are scalar values applied to each features' log-likelihood. Initialized to 1 for all if there is no weight vector passed in. The optimal weight vector was found by discretizing the possible weight values and grid-searching over all combinations.

- ```train```: Fits the classifier to the training data by computing class priors, means, variances, and conditional probabilities for both numerical and categorical features.

    1. #### Class Prior Calculation:

        -   For each class $c$, compute the prior probability $P(c)$ as: $P(c)= \frac{Total number of samples}{Number of samples in class c}$

    2. #### Categorical Feature Processing

        - For each categorical feature and class $c$, compute the conditional probability: $P(x_j = v|c) = \frac{\text{Count of } x_j = v \text{ in class } c}{N_c}$
        - Store these probabilities in a nested dictionary structure for efficient lookup during prediction.
    
    3. #### Numerical Feature Processing: 
    
        -  For each numerical feature $x_j$ and class $c$, calculate:

            - #### Mean: $\mu_{c,j} = \frac{1}{N_c}\sum_{i=1}^{N_c}x_{i,j}$

            - #### Variance: $\sigma^2_{c,j} = \frac{1}{N_c}\sum_{i=1}^{N_c}(x_{i,j} - \mu_{c,j})^2 + \epsilon$

            - #### where: 
                
                - $N_c$ is the number of samples in class $c$

                - $\epsilon = 1e^{-9}$ is a small constant added for numerical stability

- ```classify```: Predicts class labels for given feature instances by computing posterior probabilities.

    - Iterates over each instance in the feature set.
    - Calls the ```_classify_instance``` method to compute the posterior probability for each instance.  

- ```_classify_instance```: Computes the posterior probability for a single instance across all classes and selects the class with the highest probability.

    #### Mathematical Context:

    - For each class $c$, the posterior probability $P(c∣x)$ is proportional to: 
        
        $P(c|x) \propto P(c) \prod_j P(x_j|c)$

    - To prevent underflow and simplify calculations, take the natural logarithm: 
    
        $P(c∣x)$ is proportional to: $P(c|x) \propto log(P(c)) + \sum_j log(P(x_j|c))$

    #### Implementation Details:

    1. Log Prior:
        
        -  Compute $logP(c)$ using the class priors.

    2. Log-Likelihood for Numerical Features:

        - Calls ```_calculate_log_likelihood``` to compute the sum of log-likelihoods for numerical features.

    3. For each categorical feature $x_j$, compute: 
    
        $logP(x_j∣c) = log(\text{Conditional Probability from the stored dictionary})$

        - If a feature value is unseen, a small probability $\epsilon$ is used to prevent zero probability.

    4. Total Posterior Score:

        - Sum the log prior and log-likelihoods:

            $\text{Total Score} = log P(c) + \text{ log-likelihood Numerical + log-likelihood Categorical}$

    5. Prediction:

        - Select the class with the highest total posterior score.

- ```_calculate_log_likelihood```: Calculates the log-likelihood for numerical features using Gaussian probability density functions.
The architecture efficiently handles both numerical and categorical data by maintaining separate indices and processing mechanisms for each type. 

    #### Mathematical Context: 
    The Gaussian (Normal) probability density function for a single feature is given by: $PDF(𝑥) = \frac{1}{\sqrt{2\pi\sigma^2}} exp⁡(\frac{−(𝑥 − \mu)^2}{2\sigma^2})$ 

    where:
    - $𝑥$ is the data point.
    - $𝜇$ is the mean.
    - $𝜎^2$ is the variance.

    Logarithm Transformation:
    Taking the natural logarithm of the PDF simplifies multiplication into addition, which is computationally more efficient and numerically stable, especially when dealing with very small probabilities.

    $log(PDF(𝑥)) = - \frac{−(𝑥 − \mu)^2}{2\sigma^2} - \frac{1}{2}log(2\pi\sigma^2)$ 

## Preprocessing
Data preprocessing is performed by the preprocess_data function, which reads training and validation datasets from CSV files using Pandas. The preprocessing steps include:

Loading datasets and separating features and labels.
Identifying categorical features and obtaining their column indices.
Converting DataFrames to NumPy arrays for efficient numerical computations.
Ensuring consistency in feature representation between training and validation datasets.
An instance in the dataset is represented as a NumPy array, with categorical features handled via their indices for appropriate probability calculations during training and prediction.

During accuracy optimization, I found that dropping columns with numerical data that is not similar to Gaussian Distributions improved accuracy. This is likely becuase the likelihoods for numerical columns are based on the probability density function model by a Gaussian Distribution. Specifically, I dropped the column(s):

- ```min_avg5``` 

## Model Building 
The model building process involves training the Naive Bayes classifier using the processed training data: 

- Class Prior Calculation: Computes prior probabilities $P(c)$ for each class based on their frequency in the training data. 

- Statistical Computations for Numerical Features: Calculates mean $μ_{c,j}$ and variance $\sigma_{c,j}^2$ for each numerical feature $x_j$ within each class $c$.

- Conditional Probabilities for Categorical Features: Computes $P(x_j = v∣c)$ for each categorical feature and class.

- Weights: Weights [0.5, 1.0, 1.5] are applied to each log-likelihood to achieve higher accuarcy. This will limit the impact of bad indicators and increase that of good indicators.

- Data Separation: Maintains separate indices for numerical and categorical features to apply appropriate statistical methods.

## Results
The classifier was evaluated on the validation dataset after grid searching over many weight combinations, achieving an accuracy of 69.6% on the validation data set and a run time of 0.106 seconds. The running time for training and prediction was efficient due to optimized data structures and computational methods. The results demonstrate the effectiveness of the Naive Bayes approach for data fitted to gaussian distributions.

### Run-Time Complexity Analysis
#### Time Complexity: 

Training: 
- Let $N$ be the number of training samples, $F$ the number of features, $C$ the number of classes, $F_{num}$ the number of numerical features, and $F_{cat}$ the number of categorical features.

- Class Prior Calculation: $O(N)$, as it involves iterating over all training samples once.

- Numerical Feature Processing: For each class and numerical feature: $O(C×F_{num})$, since we need to compute means and variances for each numerical feature per class.

- Categorical Feature Processing: For each class and categorical feature: $O(C×F_{cat})$, as it involves counting occurrences of each categorical feature value per class.

Total Training Time Complexity: $O(N×F×C)$, considering all features and classes. 

Prediction:

- Let $M$ be the number of validation samples. For each sample, compute the posterior probability for each class:

- Per Sample Complexity: $O(C×(F_{num} + F_{cat}))$.

Total Prediction Time Complexity: $O(M×C×F)$.

#### Space Complexity:

Storage of Statistics:
- Means and Variances: $O(C×F_{num})$.

- Categorical Probabilities: $O(C × F_{cat} × V)$, where $V$ is the average number of unique values per categorical feature.

Overall Space Complexity: Linear with respect to the number of classes and features.

#### Efficiency Considerations:

- Vectorization: Utilizes NumPy's vectorized operations to reduce computational overhead.
- Efficient Data Structures: Uses dictionaries for quick lookup of categorical probabilities during prediction.
- Scalability: The algorithm scales linearly with the number of samples and features, making it suitable for large datasets.

## Challenges
Several challenges were encountered during the project:

- Handling Mixed Data Types: Integrating numerical and categorical data required distinct processing strategies. This was addressed by maintaining separate indices and implementing tailored probability calculations for each data type.
- Zero Probability Issue: Categorical features with unseen values in the validation set could lead to zero probabilities. A smoothing technique was applied by adding a small constant to probabilities to prevent this issue.
- Numerical Stability: Variance calculations for numerical features could result in zero, leading to computational errors. Adding a small constant (1e-9) to the variance ensured numerical stability during log-likelihood calculations.

By addressing these challenges, the classifier was able to perform reliably and produce accurate predictions on the validation data.

