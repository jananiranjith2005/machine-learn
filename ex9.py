import streamlit as st
import numpy as np
from numpy.linalg import lstsq
from math import ceil, pi

def lowess(x, y, f, iterations):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)], [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = lstsq(A, b, rcond=None)[0]
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

# Streamlit app
st.title("LOWESS Smoothing Example")

# Number of data points
n = st.slider("Number of data points", 10, 200, 100)

# Fraction of points to use
f = st.slider("Smoothing parameter (f)", 0.01, 1.0, 0.25)

# Number of iterations
iterations = st.slider("Number of iterations", 1, 10, 3)

# Generate data
x = np.linspace(0, 2 * pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)

# Apply LOWESS
yest = lowess(x, y, f, iterations)

# Prepare data for plotting
original_data = {"x": x, "y": y}
smoothed_data = {"x": x, "y": yest}

# Plotting with Streamlit
st.subheader("Original Data vs. LOWESS Smoothed Data")
st.line_chart(original_data, width=700, height=300)
st.line_chart(smoothed_data, width=700, height=300)

# Combined plot (overlay)
combined_data = {"Original": y, "Smoothed": yest}
st.line_chart(data=combined_data, width=700, height=300)
