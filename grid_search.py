import numpy as np
import pandas as pd
import sys
import time
import itertools
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    filename='output.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
        self.weights = None  # Initialize weights as None

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
        self._num_features = num_features  # Store number of features for weight validation
        self._unique_classes = np.unique(targets)
        total_classes = len(self._unique_classes)
        self._categorical_indices = categorical_feature_indices
        self._numerical_indices = [i for i in range(num_features) if i not in categorical_feature_indices]

        self._mean_values = np.zeros((total_classes, len(self._numerical_indices)), dtype=np.float64)
        self._variance_values = np.zeros((total_classes, len(self._numerical_indices)), dtype=np.float64)
        self._class_priors = np.zeros(total_classes, dtype=np.float64)
        self._categorical_probabilities = [{} for _ in range(total_classes)]
        
        for class_idx, class_label in enumerate(self._unique_classes):
            mask = targets == class_label
            class_data = features[mask]
            self._class_priors[class_idx] = class_data.shape[0] / float(num_samples)

            if self._numerical_indices:
                numerical_data = class_data[:, self._numerical_indices]
                self._mean_values[class_idx] = np.mean(numerical_data, axis=0)
                self._variance_values[class_idx] = np.var(numerical_data, axis=0) + 1e-9  # Add small value to prevent division by zero
        
            for feature_idx in self._categorical_indices:
                categorical_feature = class_data[:, feature_idx]
                unique_vals, counts = np.unique(categorical_feature, return_counts=True)
                probabilities = counts / counts.sum()
                prob_dict = {val: prob for val, prob in zip(unique_vals, probabilities)}
                self._categorical_probabilities[class_idx][feature_idx] = prob_dict
        
        if self.weights is None:
            self.set_weights(None)
        
    def classify(self, features: np.ndarray) -> np.ndarray:
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

            total_posterior = log_prior

            if self._numerical_indices:
                instance_numerical = instance[self._numerical_indices]
                log_likelihoods_num = self._calculate_log_likelihood(class_idx, instance_numerical)
                weights_num = self.weights[self._numerical_indices]
                log_likelihood_num = np.sum(log_likelihoods_num * weights_num)
                total_posterior += log_likelihood_num

            log_likelihood_cat = 0
            for feature_idx in self._categorical_indices:
                instance_categorical = instance[feature_idx]
                feature_probs = self._categorical_probabilities[class_idx].get(feature_idx, {})
                probability = feature_probs.get(instance_categorical, 1e-9)  # Smoothing for unseen features
                log_likelihood = np.log(probability)
                weight = self.weights[feature_idx]
                log_likelihood_cat += log_likelihood * weight

            total_posterior += log_likelihood_cat
            posterior_scores.append(total_posterior)

        return self._unique_classes[np.argmax(posterior_scores)]
    
    def _calculate_log_likelihood(self, class_idx: int, x: np.ndarray) -> np.ndarray:
        """
        Compute the log-likelihood for numerical features using Gaussian distribution.
        
        Parameters:
        - class_idx (int): Index of the class.
        - x (np.ndarray): Numerical feature vector.
        
        Returns:
        - np.ndarray: Log-likelihoods for each numerical feature.
        """
        means = self._mean_values[class_idx]
        variances = self._variance_values[class_idx]
        log_gaussian = -((x - means) ** 2) / (2 * variances) - 0.5 * np.log(2 * np.pi * variances)
        return log_gaussian

def preprocess_data(training_path: str, validation_path: str) -> Tuple[Tuple[np.ndarray, np.ndarray, List[int]], Tuple[np.ndarray, Optional[np.ndarray]]]:
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
    else:
        validation_labels = None

    train_features_df = training_df.drop('label', axis=1)
    if 'label' in validation_df.columns:
        validation_features_df = validation_df.drop('label', axis=1)
    else:
        validation_features_df = validation_df.copy()

    non_gaussian_columns = ['min_avg5']
    train_features_df = train_features_df.drop(non_gaussian_columns, axis=1, errors='ignore')
    validation_features_df = validation_features_df.drop(non_gaussian_columns, axis=1, errors='ignore')

    categorical_columns = ['team_abbreviation_home', 'team_abbreviation_away', 'season_type', 'home_wl_pre5', 'away_wl_pre5']
    categorical_indices = [train_features_df.columns.get_loc(col) for col in categorical_columns if col in train_features_df.columns]

    train_features = train_features_df.values
    validation_features = validation_features_df.values

    return (train_features, train_labels, categorical_indices), (validation_features, validation_labels)

def grid_search_weights(
    classifier: CustomNaiveBayesClassifier,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    weight_options: List[float],
    max_combinations: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Perform grid search to find the best set of feature weights.
    
    Parameters:
    - classifier (CustomNaiveBayesClassifier): Trained classifier.
    - validation_features (np.ndarray): Validation feature matrix.
    - validation_labels (np.ndarray): Validation labels.
    - weight_options (List[float]): List of possible weight values for each feature.
    - max_combinations (int, optional): Maximum number of combinations to try. Useful to limit computation.
    
    Returns:
    - Tuple[np.ndarray, float]: Best weights and corresponding accuracy.
    """
    num_features = classifier._num_features
    categorical_indices = classifier._categorical_indices
    numerical_indices = classifier._numerical_indices
    
    weight_grid = [weight_options] * num_features

    best_accuracy = -1
    best_weights = None

    all_combinations = itertools.product(*weight_grid)
    
    if max_combinations is not None:
        all_combinations = itertools.islice(all_combinations, max_combinations)

    total_combinations = 0
    for weights in all_combinations:
        total_combinations += 1
        weights = np.array(weights)
        classifier.set_weights(weights)
        predicted_labels = classifier.classify(validation_features)
        accuracy = np.mean(predicted_labels == validation_labels)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights.copy()
        
        if total_combinations % 1000 == 0:
            logging.info(f"Checked {total_combinations} combinations... Current best accuracy: {best_accuracy:.4f}")
            logging.info(f"Current best weights: {best_weights}")
    
    print(f"Total combinations checked: {total_combinations}")
    return best_weights, best_accuracy

def main():
    """
    Main execution function to train the classifier, perform grid search on weights, and evaluate accuracy.
    """
    if len(sys.argv) < 3:
        print("Usage: python script.py <training_file.csv> <validation_file.csv>")
        sys.exit(1)

    training_file = sys.argv[1]
    validation_file = sys.argv[2]

    processed_training, processed_validation = preprocess_data(training_file, validation_file)
    
    train_features, train_labels, categorical_indices = processed_training    
    validation_features, validation_labels = processed_validation   
    
    classifier = CustomNaiveBayesClassifier()
    classifier.train(train_features, train_labels, categorical_indices)
   
    weight_options = [0.5, 1.0, 1.5]
    
    max_combinations = None

    print("Starting grid search for feature weights...")
    best_weights, best_accuracy = grid_search_weights(
        classifier,
        validation_features,
        validation_labels,
        weight_options,
        max_combinations
    )
    
    print("Grid search completed.")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Weights: {best_weights}")

    classifier.set_weights(best_weights)
    predicted_labels = classifier.classify(validation_features)
    final_accuracy = np.mean(predicted_labels == validation_labels)
    print(f'Final Accuracy with Best Weights: {final_accuracy:.3f}')


if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
