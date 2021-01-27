# SVSplit
Approximate implementation of [Automatic Feature Decomposition for Single View Co-training](https://icml.cc/2011/papers/498_icmlpaper.pdf
)

*Approximate* because the constraint that all features should be used is included.

Use:

```python
from svco import SVCoTrain
import numpy as np

# For example, all the method requires is a numpy matrix
data = np.loadtxt('data.csv', delimiter=',')

# Confidence threshold (ct), epsilon value (eps), number of instances to add at each iter (to_add)
# All example values
ct = 0.8
eps = 0.5
to_add = 15

# svco_v1 is the first index set, svco_v1 is the second index set, labelled is the newly labelled data
# Labelled format: SAMPLE, TRUE LABEL, PREDICTED LABEL
svco_v1, svco_v2, labelled = SVCoTrain(data, ct, eps, to_add)

#...
```
