import streamlit as st
import numpy as np

class CandidateElimination:
    def __init__(self, num_features):
        self.num_features = num_features
        self.S = [np.array(['0'] * num_features)]  # Initialize specific boundary
        self.G = [np.array(['?'] * num_features)]  # Initialize general boundary

    def generalize(self, instance):
        new_G = []
        for g in self.G:
            new_g = np.copy(g)
            for i in range(self.num_features):
                if g[i] == '?':
                    new_g[i] = instance[i]
                elif g[i] != instance[i]:
                    new_g[i] = '?'
            new_G.append(new_g)
        return new_G

    def specialize(self, instance):
        new_S = []
        for s in self.S:
            if np.array_equal(s, instance):
                continue
            for i in range(self.num_features):
                if s[i] == '0' and instance[i] != '0':
                    new_s = np.copy(s)
                    new_s[i] = '?'
                    new_S.append(new_s)
        return new_S

    def update_boundary_sets(self, instance, target):
        if target == 'Y':
            self.S = [s for s in self.S if np.array_equal(s, instance)]
            self.G = self.generalize(instance)
        elif target == 'N':
            self.G = [g for g in self.G if not self.consistent_with(g, instance)]
            self.S = self.specialize(instance)

    def consistent_with(self, hypothesis, instance):
        for i in range(self.num_features):
            if hypothesis[i] != '?' and hypothesis[i] != instance[i]:
                return False
        return True

def main():
    st.title("Candidate Elimination Algorithm with Streamlit")

    num_features = st.number_input("Number of features:", min_value=1, step=1, value=2)

    # Initialize the CandidateElimination algorithm
    ce = CandidateElimination(num_features)

    st.write("Initial specific boundary (S):")
    st.write(ce.S)
    st.write("Initial general boundary (G):")
    st.write(ce.G)

    st.header("Update Hypotheses")

    instance = st.text_input("Instance (comma-separated values):")
    instance = instance.split(',') if instance else None

    if instance and len(instance) == num_features:
        target = st.radio("Target (Y/N):", ('Y', 'N'))
        if st.button("Update Hypotheses"):
            ce.update_boundary_sets(np.array(instance), target)
            st.write("Updated specific boundary (S):")
            st.write(ce.S)
            st.write("Updated general boundary (G):")
            st.write(ce.G)

if __name__ == "__main__":
    main()
