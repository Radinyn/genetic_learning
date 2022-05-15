# Julia module for Genetic Algorithm based AI

This module simulates Darwinian Evolution in order to train neural networks in various tasks.

## Model parameters:

### Train function
-> function that takes in one parameter - a feed function (will be provided by the module)
-> feed function takes in the input layer, returns output layer activations
-> train function should return a score (a value representing how well it did, the more, the better)

### Net sizes
-> dimensions of a neural net (input layer, hidden layers..., output layer)

### Agents per Generation
-> number of agents (neural nets) per generation

### Top percent
-> percent of population that gets to reproduce

### Mutation chance
-> chance for a single neuron to mutate

### Mutation limit
-> maximal change in the neuron in a single mutation



#### Future
-> Add saving to file
