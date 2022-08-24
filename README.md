# strom.jl

## About

Common algorithms for decision trees.

### What is implemented?
- [X] Regression tree (MSE criterion)
- [X] Classification tree (Gini or entropy criterion)
- [X] Random forests
- [ ] Visualisation of results
- [ ] GBDT regression
- [ ] GBDT classification
- [X] PCA
- [X] SVD
- [X] Gauss-Jordan
- [X] PowerIteration
- [X] InversePowerIteration

### For what you can use this library?
This project was written for pracitce and from curiosity and I wouldn't recommend
using it in real applications.
Most parts of it were extensively tested and should be working fine. 
However, I wrote it without emphasizing speed and also with very 
little knowledge of Julia, thus the code
doesn't use the language as would be appropriate.

Nonetheless, you can use it to learn about implementations of the listed
algorithms.
While working on it, I was sometimes looking for other implementations on the
Internet that would help me with debugging my own code.
Interestingly for many algorithms here you can't find a good reference.
Usually you can find implementations that are too complex and
super-duper-optimized with the main idea burried or you might find
nothing at all.

If you look for topnotch implementations of the above I would suggest looking
in [DecisionTree.jl](https://www.juliapackages.com/p/decisiontree),
[LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/), and
[XGBoost](https://juliapackages.com/p/xgboost)


### Note regarding boosted trees
The boosted trees were written with the following 
[paper](https://arxiv.org/abs/1603.02754) in mind.
However, they don't use some optimization tricks and the heruistic
for picking the best split might differ.

