# log-normalize the count data
Y = np.log10(Y + 1)  # these are the `train` individuals
X = np.log10(Y + 1)
# Make sure you have made the train,valid,test split
Y_valid = np.log10(Y_valid + 1)  # `valid` individuals
X_valid = np.log10(Y_valid + 1)

# standardize the data
from sklearn.preprocessing import StandardScaler
x_preproc = StandardScaler()
y_preproc = StandardScaler()
Xs = x_preproc.fit_transform(X)
Ys = y_preproc.fit_transform(Y)
Xs_valid = x_preproc.transform(X_valid)
Ys_valid = y_preproc.transform(Y_valid)

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LassoLarsCV

m = MultiOutputRegressor(LassoLarsCV(), n_jobs=10)

# Train the model using the training sets
m.fit(Xs, Ys)
Y_pred = m.predict(Xs_valid)
i = 10 # some random gene
plt.scatter(Y_pred[:, i], Ys_valid[:,i], alpha=0.1)

# Evaluate the performance
from scipy.stats import pearsonr, spearmanr

# Get the performance for all the genes
performances = pd.Series([spearmanr(Ys_valid[:,i], Y_pred[:, i])[0]
				         for i in range(Y_pred.shape[1])],
						 index=gene_idx)
performances.plot.hist(30)