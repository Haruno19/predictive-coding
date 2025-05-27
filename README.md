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
p(u\mid v) = f(u; g(v), \Sigma_u)
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
p(v\mid u) = \frac{p(v)p(u\mid v)}{p(u)}. \tag{1}
```
Such Bayesian approach **integrates** the information brought by **the stimulus** with the **prior knowledge** ($p(v)$), but is challenging for a simple biological system.
Since $g$ relating the variable we wish to infer with observations is non-linear, the *posterior distribution* $p(v\mid u)$ may not take a standard shape, thus requiring representing infinitely many $p(v\mid u)$ values for different possible $u$, rather than a few summary statistics like the *mean* and the *variance*. Furthermore, computing the *posterior* involves computing the *normalization term* $p(u)$, which in turn involves evaluating an integral, which is very challenging for a simple biological system.

#### 2.2 Gradient Ascent — *Finding the most likely feature value*

Instead of finding the whole *posterior distribution* $p(v\mid u)$, we’ll try to find the **most likely** value of $v$ that **maximizes** $p(v\mid u)$, denoted by $\phi$. In short, our goal is to find the value $\phi$ that maximizes $p(v\mid u)$, meaning the value $\phi$ such that $p(\phi\mid u)$ is the largest $p(v\mid u)$ between all the values of $v$.

According to $(1)$,  $p(\phi\mid u)$ depends on a **ratio** of two quantities, but the **denominator** $p(u)$ **does not depend** on $\phi$. Thus, the value of $\phi$ that maximizes $p(\phi\mid u)$ is the same that maximizes the **numerator** of $(1)$. 
We’ll call $F$ the **logarithm** of the numerator:
```math
F = \ln{p(\phi)}+\ln{p(u\mid \phi)}
```
Maximizing $F$ is the same as maximizing the numerator or $(1)$, since $\ln$ is a **monotonic** function, and its easier since the expressions for $p(\phi)$ and $p(u\mid \phi)$ involve exponentiation. 
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
\frac{\partial{F}}{\partial{\phi}}=\frac{v_p-\phi}{\Sigma_p}+\frac{u-g(\phi)}{\Sigma_u}g'(\phi). \tag{2}
```
In our example, since $g(\phi)=\phi^2$, $g'(\phi)=2\phi$.

To find our best guess $\phi$ for $v$, we can simply change $\phi$ in proportion to the gradient:
```math
\dot{\phi}=\frac{\partial{F}}{\partial{\phi}}.
```
$\dot{\phi}$ is the **rate of change** of $\phi$ over time. The first term of $(2)$ moves $\phi$ in the direction of the mean of $p(v)$ (the *prior*), and the second term moves it according to the sensory stimulus; both terms are **weighted** by the **reliabilities** of the prior and the sensory input respectively. The **variance**, respectively $\Sigma_p$ and $\Sigma_u$ for the prior knowledge and the sensory stimulus, represent the uncertainty of the respective data, thus the **inverse of the variance** represents its precision, or *reliability*, and acts as a **weight** for each respective term in the equation $(2)$. 

This method of gradient ascent is computationally much simpler than the Bayesian Inference, and quickly converges to the desired value.  

#### 2.3 Possible Neural Implementation

Let’s denote the two terms in $(2)$ as follows:
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
Once these equations converge, $\dot{\epsilon}=0$; setting $\dot{\epsilon}=0$ and solving these equations for $\epsilon$, we obtain the same equations denoting the two terms of $(2)$ described earlier.  

<img src="https://github.com/Haruno19/predictive-coding/blob/main/Attachments/PCN.fig3.png" width="60%">  
  
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

Maximizing $p(u)$ directly however, would involve working with a complicated integral. It’s simpler to maximize the *joint probability* $p(u,\phi)$ instead, since $p(u,\phi)=p(\phi)p(u\mid \phi)$, therefore $\ln{p(u,\phi)}=F$.

The intuition is, maximizing $p(u)$ would mean to integrate (and thus “average”) the following:
```math
p(u) = \int p(u, \phi) \, d\phi = \int p(u\mid \phi) p(\phi) \, d\phi
```
over all possible values $\phi$ can assume , which is computationally very intricate especially for a biological system. Instead, we select our specific inferred **best** guess $\hat\phi$ as maximize the joint probability $p(u,\hat\phi) = p(u\mid \phi)p(\phi)$ with respect to $\hat\phi$.

After inferring the most likely value of $\phi$: $\hat\phi$, given $u$ as the light intensity input and the current model parameters, we want to tune our parameters in order to maximize the probability density $p(u, \hat\phi)$. 
Basically, the animal asks itself: 
> “Given my best guess of $\phi$, how likely is it that my internal parameters would expect its light intensity to be the measured one, $u$?”. 

Intuitively, since $\hat\phi$ was guessed on the basis of $u$, this probability should be very high; however, given the inaccuracies in the model parameters, it might *not* be, and out goal is in fact to adjust the parameters in order for $p(u,\hat\phi) = p(u\mid \phi)p(\phi)$ to be maximum. 

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

The equations for the rates of change simplify significantly when rewritten in terms of the **prediction errors** $(\text{e0})$ and $(\text{e1})$ as follows:
```math
\dot v_p = \epsilon_p \tag{b0}
```
```math
\dot\Sigma_p = \frac{1}{2}(\epsilon_p^2-\Sigma_p^{-1}) \tag{b1}
```
```math
\dot\Sigma_u = \frac{1}{2}(\epsilon_u^2-\Sigma_u^{-1}) \tag{b2}
```
These parameters update rules correspond to very simple **synaptic plasticity mechanisms**, since all the rules include only values that can be known by the synapse (the activity of the pre-synaptic and post-synaptic neurons, and the strengths of the synapse itself) and are also **Hebbian**, since they depend on the products of those activities.
For example, the update rule for $v_p$ is equal to the product of the pre-synaptic activity $1$ and the post-synaptic activity $\epsilon_p$.

It’s important to note that the **minimum value** of $1$ has to be imposed on the estimated variances $\Sigma_p$ and $\Sigma_u$ to prevent $\epsilon_p$ and $\epsilon_u$ from diverging (or converging too slowly). 

#### 2.5 Learning the relationship between variables

One assumption we’ve made so far is that the function $g$ that relates the variable being inferred $\phi$ to the measured stimulus $u$ (until now, $g(v)=v^2$, meaning that, if the food size is $v$, the animal expects the measured light intensity to be equal to $g(v)$) was known. In reality, this relation **might not be known**, or might need to be **refined** and tuned too. 

From now on, we will consider a function $g(v, \theta)$ that depends also on a parameter $\theta$. 
##### First case — Linear function

First, we’ll consider a simple **linear** function $g(v,\theta) = \theta v$ where the parameter $\theta$ has a clear biological interpretation. 
Following this change in the model, the equations $(\text{a0})$, $(\text{a1})$ and $(\text{a2})$ change as follows *(Note: $g'(v,\theta)=\theta$)*:
```math
\begin{gathered}
\dot{\phi} = \theta\epsilon_u-\epsilon_p \\
\dot{\epsilon_p}=\phi-v_p-\Sigma_p\epsilon_p \\
\dot{\epsilon_u}=u-\theta\phi-\Sigma_u\epsilon_u 
\end{gathered}
```
This allows to further simplify the graph representing our model:  
<img src="https://github.com/Haruno19/predictive-coding/blob/main/Attachments/PCN.fig4a.png?raw=true" width="60%">  

The **excitatory** and **inhibitory** connections between $\epsilon_u$ and $\phi$ are significantly simplified, and can now be represented with a simple label representing the synaptic weight. In fact, node $\phi$ receives excitatory input $\theta\times\epsilon_u$, and $\epsilon_u$ receives inhibitory input $\theta\times\phi$.

After rewriting $F$ with the new definition of $g$, we can also derive the **rate of change** of $\theta$ by finding the gradient of $F$ over $\theta$:
```math
\frac{\partial F}{\partial\theta}=-\frac{2\theta\phi^2-2\phi u}{2\Sigma_u} = \frac{-2\phi(\theta\phi-u))}{2\Sigma_u}=\phi\frac{u-\theta\phi}{\Sigma_u}=\phi\frac{u-g(\phi,\theta)}{\Sigma_u}
```
which, when written in terms of $(\text{e1})$, simplifies to:
```math
\dot\theta = \epsilon_u\phi
```
It’s important to note that this rule is **Hebbian** as well, as the synaptic weights encoding $\theta$ are modified proportionally to the activities of the pre-synaptic and post-synaptic neurons $\epsilon_u$ and $\phi$. 
##### Second case — Nonlinear function

Second, we’ll consider a **nonlinear** function $g(v,\theta)=\theta h(v)$ where $h(v)$ is a nonlinear function that just depends on $v$. This results in an only slightly more complex neural implementation. 
As the first case, the equations $(\text{a0})$, $(\text{a1})$ and $(\text{a2})$ change as follows:
```math
\begin{gathered}
\dot{\phi} = \theta\epsilon_uh'(\phi)-\epsilon_p \\
\dot{\epsilon_p}=\phi-v_p-\Sigma_p\epsilon_p \\
\dot{\epsilon_u}=u-\theta h(\phi)-\Sigma_u\epsilon_u 
\end{gathered}
```
Accordingly, the graph representation of the model changes as follows:  

<img src="https://github.com/Haruno19/predictive-coding/blob/main/Attachments/PCN.fig4b.png?raw=true" width="60%">  

In this new neural implementation, the activities sent between nodes $\epsilon_u$ and $\phi$ are subject to nonlinear transformations, $h(\phi)$ and $h'(phi)$. 
A possible actual implementation of this mechanism could be achieved by adding an **additional node**, for example, between $\phi$ and $\epsilon_u$, that takes excitatory input from $\phi$, transforms it via a nonlinear transformation $h$, and then sends its output to node $\epsilon_u$ with a weight $\theta$. 
Another possible solution would be, for example, implementing the nonlinear transformation $h'$ within node $\phi$, by making it react to its input differentially depending on its level of activity. 

Analogously to the previous case, we can derive the **rate of change** of $\theta$ calculating the gradient of $F$ over $\theta$, resulting in:
```math
dot\theta = \frac{\partial F}{\partial\theta} = \epsilon_uh(\phi)
```
This rule too is **Hebbian** for the top connection between $\epsilon_u$ and $h(\phi)$. 
As for the bottom connection between nodes $\phi$ and $\epsilon_u$, it would be interesting to investigate how the rule could be implemented. Anyhow, this connection too satisfies the constraint of **local plasticity**, since $h(\phi)$ is fully determined by $\phi$. 

### Section 3 — Free-energy

In this section, we’ll discuss the relationship between the computation in the model and a technique of statistical inference involving the **minimization of free energy**.

As mentioned in the previous section, the **posterior distribution** $p(v\mid u)$ may have a complicated shape. Thus, we approximate it with a typical distribution $q(v)$, namely the **delta distribution**, which allows us to have to **infer just a single parameter** $\phi$ to describe it, instead of potentially infinitely many required to characterize a distribution of arbitrary shape.
The parameter $\phi$ represents the most likely value of $v$; the **delta distribution** $q(v)$ is equal to $0$ for all values different from $\phi$, but its integral is equal to $1$.

We want the distribution $q(v)$ to be as close as possible to the actual posterior. A measure of dissimilarity between distributions can be calculated through the **Kullback-Leibler divergence**, defined as follows:
```math
KL(q(v),p(v\mid u)) = \int q(v)\ln{\frac{q(v)}{p(v\mid u)}}dv
```
If the two distributions $q(v)$ and $p(v\mid u)$ were identical, their ratio would be equal to $1$, therefore, its logarithm would be equal to $0$, and so the whole expression would be $0$. On the other hand, the more different the two distributions are, the larger the Kullback-Leibler divergence value would be. 

Since we choose our approximate distribution to be a **delta function**, we’ll simply look for the value of its centre parameter $\phi$ that **minimizes** the Kullback-Leibler divergence. 
This minimization process might seem implausible, as it still requires to compute the term $p(v\mid u)$ involving the computation of the difficult normalization integral. However, there exists another way of finding the approximate distribution $q(v)$ that doesn’t involve complicated integral computations. 

By substituting $p(v\mid u)$ with the definition of *conditional probability* $p(v\mid u) = p(u, v)p(u)$ into the previous definition, we obtain:
```math
\begin{gathered}
KL(q(v),p(v\mid u)) = \int q(v)\ln{\frac{q(v)p(u)}{p(u, v)}}dv \\
= \int q(v)\bigg(\ln{q(v)}-\ln{p(u,v)}+\ln{p(u)}\bigg) dv\\
= \int q(v)\ln{\frac{q(v)}{p(u, v)}}dv + \int q(v)\ln{p(u)}dv 
\end{gathered} 
```
Then, since $p(u)$ is a constant —not a function of $v$— it can be brought out of the second integral. Furthermore, we note that $\int q(v)dv = 1$ since $q(v)$ is a probability distribution, thus we can further simplify the expression as follows: 
```math
\begin{gathered}
= \int q(v)\ln{\frac{q(v)}{p(u, v)}}dv + \int q(v)dv\times\ln{p(u)} \\
= \int q(v) \ln{\frac{q(v)}{p(u,v)}} dv + \ln{p(u)}
\end{gathered}
```
The integral in the last line of the above equation is called **free-energy**. We will denote its **negative** by $F$, since we’ll show that the negative free energy is equal to the function $F$ we defined and used in Section 2. 
```math
F = \int q(v) \ln{\frac{p(u,v)}{q(v)}} dv \tag{3}
```
We note that the **negative free-energy** $F$ is related to the Kullback-Leibler divergence in the following way:
```math
KL(q(v),p(v\mid u)) = -F-\ln{p(u)}.
```
$\ln{p(u)}$ does not depend on $\phi$, the parameter describing $q(v)$, so the value $\phi$ that **maximizes** $F$ is the **same value** that **minimizes** the distance between $q(v)$ and $p(v\mid u)$. 

Assuming $q(v) = \delta(v-\phi)$ is a delta distribution (which has a property that for any function $h(x)$, $\int \delta(x-\phi)h(x)=h(\phi)$), we can further simplify the negative free-energy as follows:
```math
\begin{gathered}
F = \int \delta(v-\phi) \ln{\frac{p(u,v)}{\delta(v-\phi)}} dv \\
= \int \delta(v-\phi) \ln{p(u,v)}dv - \int \delta(v-\phi)\ln{\delta(v-\phi)}dv \\
= \ln{p(u,\phi)}-\ln{\delta(\phi-\phi)} \\
= \ln{p(u,\phi)}-\ln{\delta(0)} \\
= \ln{p(u, \phi)} + C_1
\end{gathered}
```
Using $p(u,\phi) = p(\phi)p(u\mid\phi)$ (and ignoring the constant term $C_1$), we obtain the same expression for $F$ introduced in the previous section.

In Section 2.4 we discussed how we wish to find the parameters of the model for which the sensory observation $u$ is **least surprising** —in other words, those which **maximize** $p(u)$. According to $(3)$, $p(u)$ is related to the negative free energy $F$ in the following way:
```math
\ln{p(u)}=F+KL(q(v),p(v\mid u))
```
Since the Kullback-Leibler divergence is non-negative, $F$ sets a **lower bound** on $\ln{p(u)}$. So, by maximizing $F$, we can both find an approximate distribution $q(v)$, and optimize the model parameters.

### Section 4 — Scaling up the model of perception

#### 4.1 Increasing the dimension of sensory input

As the **dimensionality** of the inputs and features **increases**, the nodes’ dynamics and the synaptic plasticity rules remain the same as described in the previous sections, just generalized to multiple dimensions.

With the necessary introduction of matrices and vectors, to clarify the notation, we’ll denote single numbers or variables in *italic* ($x$), column vectors with bars ($\bar x$), and matrices in **bold** ($\bf x$). 

We assume the animal has observed sensory input $\bar u$ and estimates the most likely values $\bar\phi$ for the variables $\bar v$. We also assume it has a prior expectation that the variables $\bar v$ come from a multivariate normal distribution with mean $\bar v_p$ and covariance $\bf  \Sigma_p$, i.e. $p(\bar v) = f(\bar v; v_p, {\bf \Sigma_p})$ where:
```math
f(\bar x; \bar \mu, {\bf \Sigma}) = \frac{1}{\sqrt{(2 \pi)^N |{\bf\Sigma}|}}\exp\bigg(-\frac{1}{2}(\bar x-\bar\mu)^T{\bf\Sigma}^{-1}(\bar x-\bar\mu)\bigg).
```
where $N$ is the length of vector $\bar x$, and $|{\bf \Sigma}|$ is the determinant of matrix $\bf\Sigma$.

The probability of observing sensory input $\bar u$ given the variables $\bar v$ is given by:
```math
p(\bar u\mid\bar v) = f(\bar u; g(\bar v, {\bf\Theta}),{\bf\Sigma_u})
```
where $\bf\Theta$ is a matrix containing the parameters of the generalized function $g(\bar v,{\bf\Theta})={\bf\Theta}h(\bar v)$, where each element $i$ of $h(\bar v)$ depends on $v_i$. 

The negative free-energy $F$ can now be rewritten as follows:
```math
\begin{gathered}
F = \ln{p(\bar\phi)}+\ln{p(\bar u\mid\bar\phi)} \\
= \frac{1}{2}(-\ln|{\bf\Sigma_p}|-(\bar\phi-\bar v_p)^T{\bf\Sigma_p^{-1}}(\bar\phi-\bar v_p) \\
- \ln{|{\bf\Sigma_u}|}-(\bar u-g(\bar\phi,{\bf\Theta}))^T{\bf\Sigma_u^{-1}}(\bar u-g(\bar\phi,{\bf\Theta}))) + C
\end{gathered}
```
To calculate the vector of most likely values $\bar\phi$, we will calculate the gradient of $F$ w.r.t. $\bar\phi$ —basically the vector of the derivatives $\frac{\partial F}{\partial\phi_i}$ for $i=1$ to $N$)— denoted as $\frac{\partial F}{\partial\bar\phi}$. 
To perform such computation, we can make use of the rules for computing derivatives with vectors and **symmetric** matrices, summarized in the table below:

| Organic rule                                                                                                                | Generalization for matrices                                                                                                                                            |
| --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\frac{\partial ax^2}{\partial x}=2ax$                                                                                      | $\frac{\partial \bar x^T{\bf A}\bar x}{\partial \bar x}=2{\bf A}\bar x$                                                                                                |
| if $z = f(y)$ and $y=g(x)$, then $\frac{\partial z}{\partial x}=\frac{\partial y}{\partial x}\frac{\partial z}{\partial y}$ | if $z = f(\bar y)$ and $\bar y=g(\bar x)$, then $\frac{\partial z}{\partial\bar x}=\big(\frac{\partial\bar y}{\partial\bar x}\big)^T\frac{\partial z}{\partial\bar y}$ |
| $\frac{\partial \ln{a}}{\partial a}=\frac{1}{a}$                                                                            | $\frac{\partial \ln{\mid{\bf A}\mid}}{\partial {\bf A}}={\bf A}^{-1}$                                                                                                  |
| $\frac{\partial \frac{x^2}{a}}{\partial a}=-\frac{ x^2}{a^2}$                                                               | $\frac{\partial \bar x^T {\bf A}\bar x}{\partial {\bf A}}= -({\bf A}^{-1}\bar x)({\bf A}^{-1}\bar x)^T$                                                                |
  
Since $\bf\Sigma$ are **covariance** matrices, they are **symmetric**, so we can apply these rules to compute the gradient of the negative free-energy $F$ w.r.t. $\bar\phi$. The result is as follows:
```math
\frac{\partial F}{\partial\bar\phi}=-{\bf\Sigma_p^{-1}}(\bar\phi-\bar v_p)+\frac{\partial g(\bar\phi,{\bf\Theta})^T}{\partial\bar\phi}{\bf\Sigma_u^{-1}}(\bar u - g(\bar\phi,{\bf\Theta})). \tag{4}
```

Analogously to $(2)$, the two terms of $(4)$ are generalizations of the **prediction errors**:
```math
\bar\epsilon_p={\bf\Sigma_p^{-1}}(\bar\phi-\bar v_p) \tag{E0}
```
```math
\bar\epsilon_u={\bf\Sigma_u^{-1}}(\bar u- g(\bar\phi,{\bf\Theta})) \tag{E1}
```

To briefly recap what those variables mean *semantically*:
- $\bar\epsilon_p$ is the **prior prediction error** —how much the *inferred variable* $\bar\phi$ deviates from the *prior knowledge* $\bar v_p$, weighted by how confident we are in our prior knowledge.
- $\bar\epsilon_u$ is the **sensory prediction error** —the difference between the *actual* sensory input $\bar u$ and the *predicted* sensory input $g(\bar\phi,{\bf\Theta})$, weighted by confident we are in the measurement.

It is useful to recall that we multiply —effectively **weight**— these errors by the **inverse covariance matrices** ${\bf\Sigma^{-1}}$ rather than the covariance matrices ${\bf\Sigma}$ themselves since the latter are a measure of **uncertainty**, and we want to weight our errors with a measure of **confidence** instead. The inverse covariance matrix is in fact also called the **precision matrix**. 

Furthermore, it’s useful to recall that the function $g(\bar\phi,{\bf\Theta})$ maps the inferred variable $\bar\phi$ to its *predicted* sensory data $\hat{\bar u}$; in other words, it’s the model’s prediction of what sensory data *should look like* if the hidden variable has value equal to $\bar\phi$. 

With the **prediction errors** now defined as $(\text{E0})$ and $(\text{E1})$, we can define the **rate of change** for $\bar\phi$ as follows:
```math
\dot{\bar \phi}=-\bar\epsilon_p+\frac{\partial g(\bar\phi,{\bf\Theta})^T}{\partial\bar\phi}\bar\epsilon_u. \tag{A0} 
```
