
# Week 2 Assignment: Optimisation Algorithms for Data Analysis(CS7DS2)  
**Trinity College Dublin**  
**Student:** Ujjayant Kadian  
**Student Number:** 22330954  

## Overview
This repository contains the report and accompanying Jupyter notebook for Week 2 Assignment of *Optimisation for Machine Learning (CS7DS2)*. The assignment focuses on numerical differentiation and gradient descent techniques.  

---

## 📝 Contents of the Notebook
The notebook covers three major tasks from the assignment:

### 🟡 Part (a): Symbolic and Numerical Derivatives
- **(i) Symbolic Derivative Using `sympy`**  
  - Computes the derivative of \( y(x) = x^4 \):  
    \[
    \frac{d}{dx}(x^4) = 4x^3
    \]
- **(ii) Finite-Difference Approximation**  
  - Implements a numerical approximation of the derivative:  
    \[
    D_\delta[f](x) = \frac{f(x+\delta) - f(x)}{\delta}
    \]
  - Compares results with the exact derivative for \(\delta = 0.01\).  
- **(iii) Effect of Varying \(\delta\)**  
  - Analyzes errors from changing \(\delta\) values and plots the results on a log-log scale.  

---

### 🟠 Part (b): Gradient Descent on \( y(x) = x^4 \)
- **(i) Implementation with Fixed Step Size (\(\alpha\))**  
  - Implements a gradient descent algorithm with the update rule:  
    \[
    x_{\text{new}} = x_{\text{old}} - \alpha \times 4x^3_{\text{old}}
    \]
- **(ii) Convergence Analysis**  
  - Visualizes how \(x\) and \(y(x)\) change over iterations.  
- **(iii) Sensitivity to Initial Values and Step Size (\(\alpha\))**  
  - Tests different values of \(x_0\) and \(\alpha\).  
  - Observes convergence, divergence, and oscillations.  

---

### 🟢 Part (c): Gradient Descent on Other Functions
- **(i) \(y(x) = \gamma x^2\): Impact of \(\gamma\)**  
  - Shows that convergence depends on \(\alpha\) and \(\gamma\) with the rule:  
    \[
    x_{\text{new}} = (1 - 2\alpha \gamma) x_{\text{old}}
    \]
- **(ii) \(y(x) = \gamma|x|\): Piecewise Gradient Descent**  
  - Implements subgradient descent with piecewise derivatives.  
  - Compares behavior for different \(\gamma\).  

---

## 💻 Dependencies
Make sure you have Python and Jupyter installed, along with the following libraries:

```bash
pip install sympy numpy matplotlib pandas
```

---

## 📢 Notes
- The code is structured for clarity and contains explanatory comments.  
- Plots are labeled with titles, axis labels, and legends.  
- The notebook is designed to produce all required outputs without modification.  

---

## 📝 License
This project is for academic purposes for the CS7DS2 module at Trinity College Dublin.  
Feel free to reuse the code with proper attribution.  

---

Happy Coding! 🚀