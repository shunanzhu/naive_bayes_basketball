import numpy as np
import pandas as pd
import sys
import time
import itertools
from typing import List, Tuple, Optional

class CustomNaiveBayesClassifier:
    
    def __init__(self):
        """
        Initialize all instance variables.
        """
        self._unique_classes = None
        self._class_priors = None
        self._mean_values = None
        self._variance_values = None
        self._categorical_probabilities = None
        self._categorical_indices = []
        self._numerical_indices = []
        self.weights = None
        
    def set_weights(self, weights: Optional[np.ndarray]):
        """
        Set the feature weights.
        
        Parameters:
        - weights (np.ndarray): 1D array of weights corresponding to each feature.
        """
        if weights is not None:
            if len(weights) != self._num_features:
                raise ValueError(f"Length of weights ({len(weights)}) does not match number of features ({self._num_features}).")
            self.weights = np.array(weights)
        else:
            # If no weights provided, default to 1 for all features
            self.weights = np.ones(self._num_features)

    def train(self, features: np.ndarray, targets: np.ndarray, categorical_feature_indices: List[int]=[]):
        """
        Fit the Naive Bayes classifier to the training data.
        
        Parameters:
        - features (np.ndarray): Feature matrix.
        - targets (np.ndarray): Target labels.
        - categorical_feature_indices (list): Indices of categorical features.
        """
        num_samples, num_features = features.shape
        self._num_features = num_features
        self._unique_classes = np.unique(targets)
        total_classes = len(self._unique_classes)
        self._categorical_indices = categorical_feature_indices
        self._numerical_indices = []
        for i in range(num_features):
            if i not in categorical_feature_indices:
                self._numerical_indices.append(i)

        self._mean_values = np.zeros((total_classes, len(self._numerical_indices)), dtype=np.float64)
        self._variance_values = np.zeros((total_classes, len(self._numerical_indices)), dtype=np.float64)
        self._class_priors = np.zeros(total_classes, dtype=np.float64)
        self._categorical_probabilities = []
        for _ in range(total_classes):
            self._categorical_probabilities.append({})

        for class_idx, class_label in enumerate(self._unique_classes):
            mask = targets == class_label
            class_data = features[mask]
            self._class_priors[class_idx] = class_data.shape[0] / float(num_samples)

            if self._numerical_indices:
                numerical_data = class_data[:, self._numerical_indices]
                self._mean_values[class_idx] = np.mean(numerical_data, axis=0)
                self._variance_values[class_idx] = np.var(numerical_data, axis=0) + 1e-9
        
            for feature_idx in self._categorical_indices:
                categorical_feature = class_data[:, feature_idx]
                unique_vals, counts = np.unique(categorical_feature, return_counts=True)
                probabilities = counts / counts.sum()
                prob_dict = {}
                for val, prob in zip(unique_vals, probabilities):
                    prob_dict[val] = prob
                self._categorical_probabilities[class_idx][feature_idx] = prob_dict
                    
        if self.weights is None:
            self.set_weights(None)
                
    def classify(self, features: np.ndarray):
        """
        Predict the class labels for the given feature matrix.
        
        Parameters:
        - features (np.ndarray): Feature matrix.
        
        Returns:
        - np.ndarray: Predicted class labels.
        """
        if self.weights is None:
            raise ValueError("Feature weights have not been set. Please set weights using the `set_weights` method.")
            
        predictions = [self._classify_instance(instance) for instance in features]
        return np.array(predictions)

    def _classify_instance(self, instance: np.ndarray):
        """
        Compute the posterior probabilities for a single instance and determine the most probable class.
        
        Parameters:
        - instance (np.ndarray): Feature vector of a single instance.
        
        Returns:
        - Predicted class label. 
        """
        posterior_scores = []

        for class_idx in range(len(self._unique_classes)):
            log_prior = np.log(self._class_priors[class_idx])

            if self._numerical_indices:
                instance_numerical = instance[self._numerical_indices]
                log_likelihoods_num = self._calculate_log_likelihood(class_idx, instance_numerical)
                weights_num = self.weights[self._numerical_indices]
                log_likelihood_num = np.sum(log_likelihoods_num * weights_num)
            else:
                log_likelihood_num = 0

            log_likelihood_cat = 0
            for feature_idx in self._categorical_indices:
                instance_categorical = instance[feature_idx]
                feature_probs = self._categorical_probabilities[class_idx].get(feature_idx, {})
                probability = feature_probs.get(instance_categorical, 1e-9)
                log_likelihood = np.log(probability)
                weight = self.weights[feature_idx]
                log_likelihood_cat += log_likelihood * weight

            total_posterior = log_prior + log_likelihood_num + log_likelihood_cat
            posterior_scores.append(total_posterior)

        return self._unique_classes[np.argmax(posterior_scores)]
    
    def _calculate_log_likelihood(self, class_idx: int, x: np.ndarray):
        """
        Compute the log-likelihood for numerical features using Gaussian distribution.
        
        Parameters:
        - class_idx (int): Index of the class.
        - x (np.ndarray): Numerical feature vector.
        
        Returns:
        - np.ndarray:  log-likelihoods for each numerical features.
        """
        means = self._mean_values[class_idx]
        variances = self._variance_values[class_idx]
        log_gaussian = -((x - means) ** 2) / (2 * variances) -0.5 * np.log(2 * np.pi * variances)
        return log_gaussian


def preprocess_data(training_path, validation_path):
    """
    Read and preprocess training and validation data.
    
    Parameters:
    - training_path (str): Path to the training CSV file.
    - validation_path (str): Path to the validation CSV file.
    
    Returns:
    - tuple: Processed training and validation data.
    """
    training_df = pd.read_csv(training_path)
    validation_df = pd.read_csv(validation_path)

    train_labels = training_df['label'].values
    if 'label' in validation_df.columns:
        validation_labels = validation_df['label'].values

    train_features_df = training_df.drop('label', axis=1)
    if 'label' in validation_df.columns:
        validation_features_df = validation_df.drop('label', axis=1)
    
    
    non_gaussian_columns = ['min_avg5']
    train_features_df.drop(non_gaussian_columns, axis=1, inplace=True)
    if 'label' in validation_df.columns:
        validation_features_df.drop(non_gaussian_columns, axis=1, inplace=True)
    else:
        validation_df.drop(non_gaussian_columns, axis=1, inplace=True)

    categorical_columns = ['team_abbreviation_home', 'team_abbreviation_away', 'season_type', 'home_wl_pre5', 'away_wl_pre5']
    categorical_indices = [train_features_df.columns.get_loc(col) for col in categorical_columns]

    train_features = train_features_df.values
    if 'label' in validation_df.columns:
        validation_features = validation_features_df.values
    else:    
        validation_features = validation_df.values

    if 'label' in validation_df.columns:
        return (train_features, train_labels, categorical_indices), (validation_features, validation_labels)
    else:
        return (train_features, train_labels, categorical_indices), (validation_features, None)

def main():
    """
    Main execution function to train the classifier and evaluate its accuracy.
    """
    if len(sys.argv) < 3:
        print("Usage: python script.py <training_file.csv> <validation_file.csv>")
        sys.exit(1)

    training_file = sys.argv[1]
    validation_file = sys.argv[2]

    processed_training, processed_validation = preprocess_data(training_file, validation_file)
    
    train_features, train_labels, categorical_indices = processed_training    
    validation_features, validation_labels = processed_validation   
    
    # Local: 0.696 | Gradescope: 0.676
    best_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 1., 1.5, 1., 0.5, 1.5, 0.5, 1., 0.5]
    # Local: 0.693 | Gradescope: 0.684
    # best_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5]
    # Local: 0.691 | Gradescope: 0.679
    # best_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1., 0.5]
    # Local: 0.690 | Gradescope: 0.675
    # best_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.5, 1.5, 0.5, 1.5, 0.5]
    # Local: 0.689 | Gradescope: 0.675
    # best_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5]
    # Local: 0.685 | Gradescope: 0.679
    # best_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 1.0, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5]
    
    # best_weights = [1., 1., 1., 1.5, 0.91414087, 0.5, 0.5, 0.82280186, 0.68182954, 0.85284851, 0.6303193, 0.73024106, 0.53660812, 0.77507777, 1.21229094, 1., 0.5, 0.5, 0.72445117, 0.78370277, 0.6540276,  0.54041369, 0.52733449, 0.5, 0.5, 0.5, 0.5, 0.5, 1.]
    classifier = CustomNaiveBayesClassifier()
    classifier.train(train_features, train_labels, categorical_indices)
    
    weights = np.array(best_weights)
    classifier.set_weights(weights)
    predicted_labels = classifier.classify(validation_features)
    
    for pred in predicted_labels:
        print(pred)
    
    # accuracy_score = np.mean(predicted_labels == validation_labels)
    # print(f'Accuracy: {accuracy_score:.3f}')

   
if __name__ == '__main__':
    # start_time = time.perf_counter()
    main()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")
