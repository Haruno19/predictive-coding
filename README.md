# Predictive Coding Networks

## The Energy-Free Predictive Coding Framework
*Summarizing and rewriting [A tutorial on the free-energy framework for modelling perception and learning](http://dx.doi.org/10.1016/j.jmp.2015.11.003) by Rafal Bogacz (2015).*
### Section 1 — Introduction

Any computational model, in order to be **biologically plausible**, would need to satisfy two constraints:
- **Local computation**: each neuron performs computations only based on the activity of its input neurons and the synaptic weights associated with those neurons. 
- **Local plasticity**: the “strength” of a synapse (or how much influence a presynaptic neuron has on a postsynaptic one —*synaptic plasticity*) can change over time only based on the activity of the two neurons it connects (its *pre-synaptic* and *post-synaptic* neurons).

The basic computations a neuron performs are:
1. A **sum** of its input, **weighted** by the strengths of the relative synaptic connections
2. A **transformation** of that sum **through a function** that describes the relationship between the neuron’s total input and output (*firing-input*)

### Section 2 — Simplest example of perception

A simple organism tries to infer the size (diameter) of food based on the light intensity it observes.

In this example, we consider a problem in which the value of a **single variable** $v$ (*the size of the food*) has to be inferred from a **single observation** $u$ (*the light intensity*), and $g(v) = v^2$ is a **non-linear function** that relates the average measured quantity (average *light intensity*) with the variable value (*food’s size*). Since the sensory input is *noisy*, when the size of the food is $v$, the perceived light intensity $u$ is normally distributed with mean $g(v)$  and variance $\Sigma_u$:  
```math
p(u|v) = f(u; g(v), \Sigma_u)
```
where $f(x; \mu, \Sigma)$ denotes the density of a normal distribution with mean $\mu$ and variance $\Sigma$:
```math
f(x; \mu, \Sigma) = \frac{1}{\sqrt{2 \pi \Sigma}}\exp\bigg(-\frac{(x-\mu)^2}{2\Sigma}\bigg).
```
This means that when the food size is $v$, the perceived light $u$ is distributed around $g(v)$ following a Gaussian distribution.

The animal can refine its guess for the size $v$ by combining the sensory stimulus with the **prior knowledge** on how large the food items usually are. It expects the size to be normally distributed with mean $v_p$ and variance $\Sigma_p$:
```math
p(v) = f(v; v_p, \Sigma_p)
```
#### 2.1 Bayesian  Inference — *Finding the exact solution*

To compute how likely different sized of $v$ are, given the observed input $u$, we could use Bayes’ theorem:
```math
(⭑)\qquad p(v|u) = \frac{p(v)p(u|v)}{p(u)}.
```
Such Bayesian approach **integrates** the information brought by **the stimulus** with the **prior knowledge** ($p(v)$), but is challenging for a simple biological system.
Since $g$ relating the variable we wish to infer with observations is non-linear, the *posterior distribution* $p(v|u)$ may not take a standard shape, thus requiring representing infinitely many $p(v|u)$ values for different possible $u$, rather than a few summary statistics like the *mean* and the *variance*. Furthermore, computing the *posterior* involves computing the *normalization term* $p(u)$, which in turn involves evaluating an integral, which is very challenging for a simple biological system.

#### 2.2 Gradient Ascent — *Finding the most likely feature value*

Instead of finding the whole *posterior distribution* $p(v|u)$, we’ll try to find the **most likely** value of $v$ that **maximizes** $p(v|u)$, denoted by $\phi$. In short, our goal is to find the value $\phi$ that maximizes $p(v|u)$, meaning the value $\phi$ such that $p(\phi|u)$ is the largest $p(v|u)$ between all the values of $v$.

According to $(⭑)$,  $p(\phi|u)$ depends on a **ratio** of two quantities, but the **denominator** $p(u)$ **does not depend** on $\phi$. Thus, the value of $\phi$ that maximizes $p(\phi|u)$ is the same that maximizes the **numerator** of $(⭑)$. 
We’ll call $F$ the **logarithm** of the numerator:
```math
F = \ln{p(\phi)}+\ln{p(u|\phi)}
```
Maximizing $F$ is the same as maximizing the numerator or $(⭑)$, since $\ln$ is a **monotonic** function, and its easier since the expressions for $p(\phi)$ and $p(u|\phi)$ involve exponentiation. 
By substituting the previous equations into $F$ and expanding it, we can rewrite it as follows:
```math
\begin{gathered}
F = \ln{f(\phi; v_p, \Sigma_p)}+\ln{f(u; g(v), \Sigma_u)} \\
= \ln\bigg[\frac{1}{\sqrt{2\pi\Sigma_p}}\exp(-\frac{(\phi-v_p)^2}{2\Sigma_p})\bigg] + \ln\bigg[\frac{1}{\sqrt{2\pi\Sigma_u}}\exp\frac{(u-g(\phi))^2}{2\Sigma_u}\bigg] \\
= \ln\frac{1}{\sqrt{2\pi}} - \frac{1}{2}\ln{\Sigma_p}-\frac{(\phi-v_p)^2}{2\Sigma_p} + \ln\frac{1}{\sqrt{2\pi}} - \frac{1}{2}\ln{\Sigma_u}-\frac{(u-g(\phi))^2}{2\Sigma_u} \\
= 2\ln\frac{1}{\sqrt{2\pi}} +\bigg( -\frac{1}{2}\ln{\Sigma_p}-\frac{1}{2}\frac{(\phi-v_p)^2}{\Sigma_p} - \frac{1}{2}\ln{\Sigma_u}-\frac{1}{2}\frac{(u-g(\phi))^2}{\Sigma_u}\bigg) 
\end{gathered}
```
Incorporating the constant term into a constant $C$ and gathering by $\frac{1}{2}$ we obtain:
```math
F = \frac{1}{2}\bigg(-\ln{\Sigma_p}-\frac{(\phi-v_p)^2}{\Sigma_p}-\ln{\Sigma_u}-\frac{(u-g(\phi))^2}{\Sigma_u}\bigg)+C
```

To find $\phi$, we’ll use **gradient ascent**, that is, we’ll modify $\phi$ proportionally to the gradient of $F$. 

The **derivative** of $F$ w.r.t. $\phi$ is:
```math
(*)\qquad \frac{\partial{F}}{\partial{\phi}}=\frac{v_p-\phi}{\Sigma_p}+\frac{u-g(\phi)}{\Sigma_u}g'(\phi).
```
In our example, since $g(\phi)=\phi^2$, $g'(\phi)=2\phi$.

To find our best guess $\phi$ for $v$, we can simply change $\phi$ in proportion to the gradient:
```math
\dot{\phi}=\frac{\partial{F}}{\partial{\phi}}.
```
$\dot{\phi}$ is the **rate of change** of $\phi$ over time. The first term of $(*)$ moves $\phi$ in the direction of the mean of $p(v)$ (the *prior*), and the second term moves it according to the sensory stimulus; both terms are **weighted** by the **reliabilities** of the prior and the sensory input respectively. The **variance**, respectively $\Sigma_p$ and $\Sigma_u$ for the prior knowledge and the sensory stimulus, represent the uncertainty of the respective data, thus the **inverse of the variance** represents its precision, or *reliability*, and acts as a **weight** for each respective term in the equation $(*)$. 

This method of gradient ascent is computationally much simpler than the Bayesian Inference, and quickly converges to the desired value.  

#### 2.3 Possible Neural Implementation

