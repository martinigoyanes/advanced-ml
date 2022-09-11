Covers the problems solved individually in the Advanced ML course taught in the [KTH's MSc Machine Learning](https://www.kth.se/en/studies/master/machine-learning/)
- [Assignment 1](assignment1/report.pdf): Dimensionality Reduction
	- PCA: 
		- Explain why this data-centering step is required while performing PCA. What
could be an undesirable effect if we perform PCA on non-centered data?
		-  Does a single SVD operation is sufficient to perform PCA both on the rows and the columns of a data matrix?
		- Explain why the use of the pseudo-inverse is a good choice to obtain the inverse
mapping of the linear map.
		- Derive PCA using the criterion of variance maximization and show that one gets the same result as with the criterion of minimizing the reconstruction error. Show this result for projecting the data into k dimensions, not just 1 dimension.
	- Multi-dimensional Scaling and Isomap:
		- Explain in English what is the intuitive reason that the “double centering trick” is necessary in order to be able to solve for S given 
		- Argue that although the solution obtained by the “first point trick” will be different than the solution obtained by the “double centering trick”, both solutions are correct
		- Show that the two methods, i.e., classical MDS when Y is known and PCA on Y, are equivalent. Which of the two methods is more efficient? (Hint: Your answer may involve a case analysis.)
		- Argue that the process to obtain the neighborhood graph G in the Isomap method may yield a disconnected graph. Provide an example. Explain why this is problematic.
		- Propose a heuristic to patch the problem arising in the case of a disconnected neighborhood graph.
		- Apply classical MDS to compute an (x, y) coordinate for each city in your dataset, given the distance matrix D. Plot the cities on a plane using the coordinates you computed.
	- PCA vs Johnson-Lindenstrauss random projections
		- Provide a qualitative comparison (short discussion) between the two methods, PCA vs. Johnson-Lindenstrauss random projections, in terms of (i) projection error; (ii) computational efficiency; and (iii) intended use cases

- [Assignment 2](assignment2/report.pdf): Probabilistic Graphical Models, Variational Inference and EM algorithms
	- Tree Graphical Model:
		- Implement a dynamic programming algorithm that for a given rooted binary T, theta and beta, computes the likelihood of the Tree Graphical Model
	- Simple Variational Inference:
		- Implement the VI algorithm for the variational distribution in Equation (10.24) in Bishop.
		- What is the exact posterior?
		- Compare the inferred variational distribution with the exact posterior. Run the inference on data points drawn from i.i.d. Gaussians. Do this for three interesting cases and visualize the results. 
	- EM algorithm for seismographic data
		- We have seismographic from an area with frequent earthquakes emanating from K super epicentra. Each super epicentra is modeled by a 2-dimensional Gaussian determining the location of an earthquake and a Poisson distribution determining its strength. Derive an EM algorithm for the model given.
		- Implement it
		- Apply it to the data provided, give an account of the success, and provide visualizations for a couple of examples. Repeat it for a few different choices of K.

- [Assignment 3](assignment3/report.pdf): Random Projections, Graph Theory, Variational Inference, EM algorithms, Hidden Markov Models
	- Success probability in the Johnson-Lindenstrauss lemma:
		- Show that O(n) independent trials are sufficient for the probability of success to be at least 95%. An independent trial here refers to generating a new projection of the data points with a newly generated projection matrix.
	- Node similarity for representation learning:
		- Explain the intuition for the definition of the given similarity measure S.
		- Show that S can be computed efficiently using matrix addition, and a single matrix inversion operation, while avoiding computing an infinite series
	- Tree graphical model with leaky units:
		- Provide a linear time algorithm that computes p(X|T, M, sigma, alpha, pi) when given a tree T with leaky units
	- Variational Inference for seismographic data:
		- Derive a VI algorithm that estimates the posterior distribution for the given model.
	- Casino Model with Hidden Markov Chains:
		- Provide a drawing of the Casino model as a graphical model. It should have a variable indicating the table visited in the k − th step, variables for all the dice outcomes, variables for the sums, and plate notation should be used to clarify that N players are involved.
		- Implement it
		- Provide data generated using at least three different sets of categorical dice distributions – what does it look like for all unbiased dice, i.e., uniform distributions, for example, or if some are biased in the same way, or if some are unbiased and there are two different groups of biased dice
		- Describe an algorithm that, given (1) the parameters Θ of the full casino model (so, Θ is all the categorical distributions corresponding to all the dice), (2) a sequence of tables r_1 . . . , r_K (that is, r_i is t_i or t′_i), and (3) an observation of dice sums s_1, . . . , s_K, outputs p(r_1, . . . , r_K | s_1, . . . , s_K, Θ).
		- You should also show how to sample r_1, . . . , r_K from p (R_1, . . . , R_K | s_1, . . . , s_K, Θ) as well as implement and show test runs of this algorithm
	- Casino Model with Expectation Maximization:
		- Present the algorithm written down in a formal manner (using both text and mathematical notation, but not pseudo code)
		- Implement it and test the implementation with data generated in Task 3.5, and provide graphs or tables of the results of testing it with the data.