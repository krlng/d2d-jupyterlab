from sklearn.metrics import precision_recall_curve
import sys
import sklearn.metrics as metrics

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
    sys.stderr.write('python evaluate.py model test metric\n\n')
    sys.stderr.write('\tmodel \t model pickle file to evaluate\n')
    sys.stderr.write('\ttest \t test data set, pickle file with features and labels\n')
    sys.stderr.write('\tmetric \t an output file with an AUC valuer\n')
    sys.exit(1)


if len(sys.argv) != 4:
    usage('Wrong number of arguments. Usage:')

model_file = sys.argv[1]
test_matrix_file = sys.argv[2]
metrics_file = sys.argv[3]

with open(model_file, 'rb') as fd:
    model = pickle.load(fd)

with open(test_matrix_file, 'rb') as fd:
    matrix = pickle.load(fd)

labels = matrix[:, 1].toarray()
x = matrix[:, 2:]

predictions_by_class = model.predict_proba(x)
predictions = predictions_by_class[:, 1]

precision, recall, thresholds = precision_recall_curve(labels, predictions)

auc = metrics.auc(recall, precision)

with open(metrics_file, 'w') as fd:
    fd.write('AUC: {:4f}\n'.format(auc))

