import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys

try:
    import cPickle as pickle   # python2
except ModuleNotFoundError:
    import pickle           # python3

try:  # python2
    reload(sys)
    sys.setdefaultencoding('utf-8')
except NameError:
    pass


def usage(msg):
    if msg:
        sys.stderr.write('{}\n\n'.format(msg))
    sys.stderr.write('python train_model.py features seed model\n\n')
    sys.stderr.write('\tfeatures \t input features and labels pickle file.\n')
    sys.stderr.write('\tseed \t\t random state (integer). Example: 20170423\n')
    sys.stderr.write('\tmodel \t\t output model pickle file.\n')
    sys.exit(1)


if len(sys.argv) != 4:
    usage('Wrong number of arguments. Usage:')


input = sys.argv[1]
output = sys.argv[3]
seed = int(sys.argv[2])

with open(input, 'rb') as fd:
    matrix = pickle.load(fd)

labels = np.squeeze(matrix[:, 1].toarray())
x = matrix[:, 2:]

sys.stderr.write('Input matrix size {}\n'.format(matrix.shape))
sys.stderr.write('X matrix size {}\n'.format(x.shape))
sys.stderr.write('Y matrix size {}\n'.format(labels.shape))

clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=seed)
clf.fit(x, labels)

with open(output, 'wb') as fd:
    pickle.dump(clf, fd)

