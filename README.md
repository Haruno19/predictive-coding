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
p(v|u) = \frac{p(v)p(u|v)}{p(u)}. \tag{⭑}
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
\frac{\partial{F}}{\partial{\phi}}=\frac{v_p-\phi}{\Sigma_p}+\frac{u-g(\phi)}{\Sigma_u}g'(\phi). \tag{∗}
```
In our example, since $g(\phi)=\phi^2$, $g'(\phi)=2\phi$.

To find our best guess $\phi$ for $v$, we can simply change $\phi$ in proportion to the gradient:
```math
\dot{\phi}=\frac{\partial{F}}{\partial{\phi}}.
```
$\dot{\phi}$ is the **rate of change** of $\phi$ over time. The first term of $(*)$ moves $\phi$ in the direction of the mean of $p(v)$ (the *prior*), and the second term moves it according to the sensory stimulus; both terms are **weighted** by the **reliabilities** of the prior and the sensory input respectively. The **variance**, respectively $\Sigma_p$ and $\Sigma_u$ for the prior knowledge and the sensory stimulus, represent the uncertainty of the respective data, thus the **inverse of the variance** represents its precision, or *reliability*, and acts as a **weight** for each respective term in the equation $(*)$. 

This method of gradient ascent is computationally much simpler than the Bayesian Inference, and quickly converges to the desired value.  

#### 2.3 Possible Neural Implementation

Let’s denote the two terms in $(*)$ as follows:
```math
\epsilon_p = \frac{v_p-\phi}{\Sigma_p} \tag{e0}
```
```math
\epsilon_u = \frac{u-g(\phi)}{\Sigma_u} \tag{e1}
```
The above terms are the prediction errors:
- $\epsilon_u$ denotes how the interred size differs from the prior expectations.
- $\epsilon_u$ represents how much the measured light intensity differs from the expected one give $\phi$ as the food item size.

Rewriting the equation for updating $\phi$ we obtain:
```math
\dot{\phi} = \epsilon_ug'(\phi)-\epsilon_p \tag{a0}
```
The model parameters $v_p$, $\Sigma_p$ and $\Sigma_u$ are assumed to be encoded in the strengths of the **synaptic connections**, while variables $\phi$, $\epsilon_p$ and $\epsilon_u$, as well as the sensory inputs, are maintained in the activity of neurons or populations of them. 

We’ll consider simple **neural nodes** which change their activity proportionally to the input they receive; for example, $(a_0)$ is implemented in the model by a node receiving input equal to the right hand of the equation. 
The nodes can compute the prediction errors with the following *dynamics*:
```math
\dot{\epsilon_p}=\phi-v_p-\Sigma_p\epsilon_p \tag{a1}
```
```math
\dot{\epsilon_u}=u-g(\phi)-\Sigma_u\epsilon_u \tag{a2}
```
Once these equations converge, $\dot{\epsilon}=0$; setting $\dot{\epsilon}=0$ and solving these equations for $\epsilon$, we obtain the same equations denoting the two terms of $(*)$ described earlier.

![[PCN.fig3.png]]
In this architecture, the computations are performed as follows.
The node $\epsilon_p$ receives **excitatory input** from node $\phi$, **inhibitory input** from a **tonically active** node $1$ with strength $v_p$, and **inhibitory input** from itself with strength $\Sigma_p$, implementing $(\text{a1})$. The nodes $\phi$ and $\epsilon_u$ analogously implement $(\text{a0})$ and $(\text{a2})$ respectively, but the information exchange between them is affected by function $g$. 

Terminology:
- an **excitatory** input is an input that **adds** to a neuron’s activity—a *positive* term.
- an **inhibitory** input is an input that **subtracts** from a neuron’s activity—a *negative* term.
- a **tonically active** node is a node that has a **constant output**—a constant in the system.
##### Breaking down the graph
To understand what the above graph represents, and how it can be used to simulate calculations, let’s start by laying down its components.
###### Nodes
The above architecture contains, first of all, **nodes**. Each node can represent either a **variable** or a **constant** (*tonically active* nodes). Every **variable** in the graph is updated in each **unit of time** to a new value, by an amount (**rate of change**) defined by the equations described by the graph itself. 

In the above architecture we have:

| Variable Nodes | Constant Nodes |
| -------------- | -------------- |
| $\epsilon_p$   | $u$            |
| $\epsilon_u$   | $1$            |
| $\phi$         |                |

###### Connections
The **lines** connecting nodes represent the **inputs** and **outputs** of each node. The direction in which the information is flowing is determined by where the point of the line is —e.g. the connection between nodes $1$ and $\epsilon_p$ has a black dot on $\epsilon_p$’s end, meaning the information flows *from* $1$ *to* $\epsilon_p$. 

There are **two types** of inputs, **excitatory** and **inhibitory**, denoted by a **arrow head** and a **black dot** respectively. The type of the input determines its **sign** in the equation described in each node, **positive** if the input is **excitatory**, and **negative** if it is **inhibitory**.

Furthermore, each connection has a **weight**, denoted by the label near each line. The **weight** is a scalar value associated with the relative input in the equation of the recipient node.
When the label is surrounded by a *box*, it signifies that  the expression, rather than being the **weight**, is the entire finale *term* of the equation. 

Summarizing the connections present in the above architecture, we have:

| Sender       | Recipient    | Sign | Weight     | Term                          |
| ------------ | ------------ | ---- | ---------- | ----------------------------- |
| $u$          | $\epsilon_u$ | $+$  | $1$        | $+1\times u$                  |
| $\epsilon_u$ | $\epsilon_u$ | $-$  | $\Sigma_u$ | $-\Sigma_u \times \epsilon_u$ |
| $\epsilon_u$ | $\phi$       | $+$  | \          | $+\epsilon_ug'(\phi)$         |
| $\phi$       | $\epsilon_u$ | $-$  | \          | $-g(\phi)$                    |
| $\phi$       | $\epsilon_p$ | $+$  | $1$        | $+1\times\phi$                |
| $\epsilon_p$ | $\epsilon_p$ | $-$  | $\Sigma_p$ | $-\Sigma_p \times \epsilon_p$ |
| $\epsilon_p$ | $\phi$       | $-$  | $1$        | $-1\times\epsilon_u$          |
| $1$          | $\epsilon_p$ | $-$  | $v_p$      | $-1\times v_p$                |
###### Interpretation and Computation
Ultimately, we can derive the equations for the **rates of change** of each **variable node**, by summing all the terms described by their input **connections** with other nodes. 
For example, the **rate of change** of node $\epsilon_p$ is:
```math
\dot{\epsilon_p}=+1\times\phi-1\times v_p-\Sigma_p\epsilon_p \\
```
Exactly equivalent to $(a_1)$. 
The same is true for the rates of nodes $\phi$ and $\epsilon_u$, respectively $(a_0)$ and $(a_2)$. 
*(Note: nodes $u$ and $1$ are **tonically active nodes**, so the value they hold doesn’t change over time, and therefore have no **rate of change**.)*

To simulate the model described by this graph then, simply means to: $[1]$ initialize the variables and constants to the desired values, $[2]$ evaluate the three rates of change $\dot{\phi}$, $\dot{\epsilon_p}$ and $\dot{\epsilon_u}$ with the derived equations, and $[3]$ update the values of the relative variable nodes, $\phi$, $\epsilon_p$ and $\epsilon_u$ respectively, with Euler’s method as follows:
```math
\begin{gathered}
\phi^{t+\Delta t} &= \phi^t + \Delta t \cdot \dot{\phi} \\
\epsilon_{p}^{t+\Delta t} &= \epsilon_{p}^{t} + \Delta t \cdot \dot{\epsilon}_p \\
\epsilon_{u}^{t+\Delta t} &= \epsilon_{u}^{t} + \Delta t \cdot \dot{\epsilon}_u
\end{gathered}
```
where $\Delta t$ is an arbitrary time step (e.g. $\Delta t = 0.01$.)

#### 2.4 Learning the model’s parameters

Our imaginary animal might wish to refine its expectation about the typical food size and the error it makes when observing light after each stimulus. In practice, we want to update the parameters $v_p$, $\Sigma_p$ and $\Sigma_u$ to gradually refine to better reflect reality. That is, to choose the parameters for which the perceived light intensities $u$ are **least surprising**: the parameters that maximize $p(u)$. 

Maximizing $p(u)$ directly however, would involve working with a complicated integral. It’s simpler to maximize the *joint probability* $p(u,\phi)$ instead, since $p(u,\phi)=p(\phi)p(u|\phi)$, therefore $\ln{p(u,\phi)}=F$.

The intuition is, maximizing $p(u)$ would mean to integrate (and thus “average”) the following:
```math
p(u) = \int p(u, \phi) \, d\phi = \int p(u|\phi) p(\phi) \, d\phi
```
over all possible values $\phi$ can assume , which is computationally very intricate especially for a biological system. Instead, we select our specific inferred **best** guess $\hat\phi$ as maximize the joint probability $p(u,\hat\phi) = p(u|\phi)p(\phi)$ with respect to $\hat\phi$.

After inferring the most likely value of $\phi$: $\hat\phi$, given $u$ as the light intensity input and the current model parameters, we want to tune our parameters in order to maximize the probability density $p(u, \hat\phi)$. 
Basically, the animal asks itself: 
> “Given my best guess of $\phi$, how likely is it that my internal parameters would expect its light intensity to be the measured one, $u$?”.

Intuitively, one might think that, since $\hat\phi$ was guessed on the basis of $u$, this probability should be very high; however, given the inaccuracies in the model parameters, it might *not* be, and out goal is to adjust the parameters in order for $p(u,\hat\phi) = p(u|\phi)p(\phi)$ to be maximum. 

In the same way we adjusted our guess of $\phi$ proportionally to the gradient of $F$, the model parameters $v_p$, $\Sigma_p$ and $\Sigma_u$ can also be optimized by adjusting them proportionally to the gradient of $F$.
In particular, we need the partial derivatives of $F$ over each one of them:
```math
\begin{gathered}
\dot v_p = \frac{\partial{F}}{\partial{v_p}}= \frac{\phi-v_p}{\Sigma_p}\\
\dot\Sigma_p =\frac{\partial{F}}{\partial{\Sigma_p}}= \frac{1}{2}\bigg( \frac{(\phi-v_p)^2}{\Sigma_p^2}-\frac{1}{\Sigma_p}\bigg)\\
\dot\Sigma_u =\frac{\partial{F}}{\partial{\Sigma_u}}= \frac{1}{2}\bigg( \frac{(u-g(\phi))^2}{\Sigma_u^2}-\frac{1}{\Sigma_u}\bigg)
\end{gathered}
```

Although the environment is **constantly variable** —food items have all different sizes, so there’s no single *ground truth* the model could possibly converge to— it’s nevertheless useful to consider the values of the “*ideal*” parameters, for which the relative **rates of change** are equal to $0$. 

For example, the expected rate of change for $v_p$ is $0$ when $\langle\frac{\phi-v_p}{\Sigma_p}\rangle = 0$ (where $\langle\rangle$ denotes the *average* over the many inferred values of $\phi$). This will happen if $v_p  \langle\phi\rangle$, i.e. when $v_p$ —the animal’s **prior knowledge** on the average food size— is indeed equal to the average expected value of $\phi$. 

Analogously, the expected rate of change for $\Sigma_p$ is $0$ when $\Sigma_p=\langle(\phi-v_p)^2\rangle$, thus when the variance of the average food size based on the animal’s prior knowledge $\Sigma_p$ is indeed equal to the average variance of $\phi$. An analogous analysys can be done for $\Sigma_u$. 

The equations for the rates of change simply significantly when rewritten in terms of the **prediction errors** $(\text{e0})$ and $(\text{e1})$ as follows:
```math
\begin{gathered}
\dot v_p = \epsilon_p \\
\dot\Sigma_p = \frac{1}{2}(\epsilon_p^2-\Sigma_p^{-1})\\
\dot\Sigma_u = \frac{1}{2}(\epsilon_u^2-\Sigma_u^{-1})\\
\end{gathered}
```
