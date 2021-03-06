# Introduction

In this project, we consider a simplified airline revenue management task using bid-price control. Bid-price control is a common example of revenue management, a common problem in operations research. Companies must delicately balance accepting arriving bids to earn revenue immediately, while also conserving resources for future bids. The structure of this problem makes it particularly suitable to model as a Markov Decision Process.

As a motivating example, consider a scenario involving a fictional airline company called Ithaca Flies. Ithaca Flies only operates three direct flights: Ithaca to NEK, Ithaca to Detroit, and Detroit to LA. Additionally, one may elect to fly from Ithaca to LA via Detroit. Each flight has economy and business seats. Ithaca Flies implements a bid-price control policy to sell tickets on these flights, where the bid-prices are the fixed trip fares. The goal of the controller is to maximize revenue. Previous approaches to this task used linear programming, while we aim to use deep reinforcement learning. In particular, we use the popular Advantage Actor Critic (A2C) algorithm using three different sets of neural network architectures. We hope this approach will serve as a more flexible approach to the task of bid-price control for airline revenue management. 

# Problem Formulation

We follow the formulation described in \cite{adelman2007dynamic}, wherein this problem is framed as a finite-horizon Markov Decision Process (MDP). In that context, we consider a scenario in which the relevant airline has a central hub, and $L$ other locations. There are single-leg itineraries between the hub and each location in both directions, as well as two-leg itineraries between different locations. Therefore in total, we have m=2L distinct flights, also called resources. As we additionally distinguish between low and high class fares for each itinerary, we therefore have n = 2(m+L(L-1)) distinct itineraries, which also characterizes the type of each customer.

As part of this formulation, we also specify a matrix A, where the ijth element which denotes the amount of resource i a customer of type j uses, i.e. a two-leg itinerary requires a seat on each leg.

Finally, we specify for each class and time step t, a probability p_{t,j} which is the probability that a customer of type j arrives at time t. Note we restrict that only one customer arrives at each period.

State space: non-negative integer for each resource. Initialized at a specified capacity (i.e. the number of seats on each flight).

Action space: (0,1) for each class j, denoting whether controller would accept a class j customer if one arrives. We enforce that there are sufficient resources if a_j > 0 (according to the A matrix).

Rewards: revenue per class, specified by user.

Transitions: given state s and action a, with probability p_{t,j} the next state will be s - A^ja_j

# Usage

The test.ipynb Jupyter notebook contains all of the code required to run on the Airline environment. It already imports the custom environment as well as our custom policy classes. It should be self-evident which blocks to run to vary the network architecture.

# Dependencies
Python 3.8.5, stable-baselines3 1.0, gym 0.18.0, numpy 1.20.1, torch 1.8.1, jupyterlab 3.0.14.

Note that on Intel Macs, you may need to also run "pip install nomkl" as the Intel Math Kernel Library seems to not play nicely with pytorch.


[1]. Adelman, D. (2007). Dynamic bid prices in revenue management. Operations Research, 55(4), 647-661.
