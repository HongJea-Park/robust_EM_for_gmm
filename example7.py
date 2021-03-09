import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Although Axes3D is not used directly,
# it is imported because it is needed for 3d projection.

import robustEM
from data.generator import Generator_Multivariate_Normal


means = [[-5, -10, 0],
         [0, -10, 0],
         [5, -10, 0],
         [-5, -0, 4],
         [0, -0, 4],
         [5, -0, 4],
         [-5, 10, 8],
         [0, 10, 8],
         [5, 10, 8]]

covs = [[[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[.5, 0, 0], [0, 8, 0], [0, 0, 2]],
        [[1, 0, 0], [0, 8, 0], [0, 0, 2]],
        [[.5, 0, 0], [0, 8, 0], [0, 0, 2]],
        [[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[.5, 0, 0], [0, 2, 0], [0, 0, 1]]]

mix_prob = [1/16, 2/16, 1/16, 2/16, 4/16, 2/16, 1/16, 2/16, 1/16]

ex7 = Generator_Multivariate_Normal(means, covs, mix_prob)
X = ex7.get_sample(1600)

# robustEM
rem = robustEM.RobustEM()
rem.fit(X)
record = rem.save_record(save_option=True, filename='example7')

c_pred = rem.predict(X)
c_list = np.unique(c_pred)

# visualization
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], color='g')
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=c_pred)
ax1.set_title('The 3-dimensional data set with nine clusters')
ax2.set_title('Clustering results by the robust EM algorithm')
plt.savefig('../plot/example7.png', dpi=300)
plt.show()
