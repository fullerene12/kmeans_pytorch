pytorch implementation of basic kmeans algorithm(lloyd method with forgy initialization) with gpu support
Adapted from [overshiki](https://github.com/overshiki/kmeans_pytorch)

usage:
```python
from kmeans_pytorch.kmeans import lloyd
import numpy as np 

A = np.concatenate([np.random.randn(1000, 2), p.random.randn(1000, 2)+3, p.random.randn(1000, 2)+6], axis=0)

#lloyd(X, n_clusters = 3, device=0, epoch = 1)
clusters_index, centers = lloyd(A, n_clusters=2, device=0, epoch=10)
```

For handling very large dataset, you can use batch processing by calling
```python
from kmeans_pytorch.kmeans import lloyd_batch
#lloyd_batch(X, n_clusters = 3, device=0, epoch = 1, batch_size=100)
clusters_index, centers = lloyd(A, n_clusters=2, device=0, epoch=10,batch_size=100)
```

See kmeans_test.ipynb for some test examples

![](./single.png)
![](./triple.png)