# ORIE6590-Airline

We are investigating Airline Revenue Management using Deep RL. We focus on a scenario involving a fictional airline company called Ithaca Flies which operates three flights: Ithaca to NEK, Ithaca to Detroit, and Detroit to LA. We follow the formulation described in [1]. In particular, we have 6 resources and 8 classes, where resources are economy and business for the three flights, and the classes are economy and business for the three direct flights as well as the Ithaca to LA via Detroit flight.

As part of this formulation, we also specify a matrix A, where the ijth element which denotes the amount of resource i a customer of type j uses, e.g. a business seat on the Ithaca to LA via Detroit flight requires a business seat on both flights.

Finally, we specify for each class and time step t, a probability p_{t,j} which is the probability that a customer of type j arrives at time t. Note we restrict that only one customer arrives at each period.

State space: non-negative integer for each resource. Initialized at a specified capacity (i.e. the number of seats on each flight).

Action space: (0,1) for each class j, denoting whether controller would accept a class j customer if one arrives. We enforce that there are sufficient resources if a_j > 0 (according to the A matrix).

Rewards: revenue per class, specified by user.

Transitions: given state s and action a, with probability p_{t,j} the next state will be s - A^ja_j


[1]. Adelman, D. (2007). Dynamic bid prices in revenue management. Operations Research, 55(4), 647-661.
