import numpy as np

class CandidateElimination:
    def _init_(self, num_features):
        self.num_features = num_features
        self.specific_h = np.array([[None] * num_features])
        self.general_h = np.array([['?' for _ in range(num_features)]])
        
    def fit(self, X, y):
        for i, x in enumerate(X):
            if y[i] == 1:  # Positive example
                self.specific_h = self._generalize_specific_h(x)
                self.general_h = self._remove_inconsistent_general_h(x)
            else:  # Negative example
                self.general_h = np.vstack((self.general_h, self._specialize_general_h(x)))

    def _generalize_specific_h(self, x):
        for i in range(len(self.specific_h[0])):
            if self.specific_h[0][i] is None or self.specific_h[0][i] == x[i]:
                continue
            else:
                self.specific_h[0][i] = '?'
        return self.specific_h

    def _remove_inconsistent_general_h(self, x):
        consistent_general_h = []
        for g in self.general_h:
            if all(g[i] == '?' or g[i] == x[i] for i in range(len(g))):
                consistent_general_h.append(g)
        return np.array(consistent_general_h)

    def _specialize_general_h(self, x):
        possible_h = []
        for g in self.general_h:
            new_g = np.copy(g)
            for i in range(len(x)):
                if g[i] == '?':
                    new_g[i] = x[i]
            possible_h.append(new_g)
        return possible_h

# Get user input for the number of features
num_features = int(input("Enter the number of features: "))
# Initialize the CandidateElimination object
ce = CandidateElimination(num_features)

# Get user input for training data X and labels y
X = []
y = []
num_examples = int(input("Enter the number of training examples: "))
print("Enter training examples (each example as a row with binary features, followed by the label):")
for _ in range(num_examples):
    example = list(map(int, input().split()))
    X.append(example[:-1])
    y.append(example[-1])
X = np.array(X)
y = np.array(y)

# Fit the model
ce.fit(X, y)
