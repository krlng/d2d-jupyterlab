import sys
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

np.set_printoptions(suppress=True)

try:
    import cPickle as pickle   # python2
except ModuleNotFoundError:
    import pickle           # python3

try:  # python2
    reload(sys)
    sys.setdefaultencoding('utf-8')
except NameError:
    pass

np.set_printoptions(suppress=True)


def usage(msg):
    if msg:
        sys.stderr.write('{}\n\n'.format(msg))
    sys.stderr.write('python featurization.py data-train data-test features-train features-test\n\n')
    sys.stderr.write('\tdata-train \t input train data set.\n')
    sys.stderr.write('\tdata-test \t input test data set.\n')
    sys.stderr.write('\tfeatures-train \t output train features matrix pkl file.\n')
    sys.stderr.write('\tfeatures-test \t output test features matrix pkl file.\n')
    sys.exit(1)


if len(sys.argv) != 5:
    usage('Wrong number of arguments. Usage:')

train_input = sys.argv[1]
test_input = sys.argv[2]
train_output = sys.argv[3]
test_output = sys.argv[4]


def get_df(input):
    df = pd.read_csv(
        input,
        encoding='utf-8',
        header=None,
        delimiter='\t',
        names=['id', 'label', 'text']
    )
    sys.stderr.write('The input data frame {} size is {}\n'.format(input, df.shape))
    return df


def save_matrix(df, matrix, output):
    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([id_matrix, label_matrix, matrix], format='csr')

    msg = 'The output matrix {} size is {} and data type is {}\n'
    sys.stderr.write(msg.format(output, result.shape, result.dtype))

    with open(output, 'wb') as fd:
        pickle.dump(result, fd, pickle.HIGHEST_PROTOCOL)


df_train = get_df(train_input)
train_words = np.array(df_train.text.str.lower().values.astype('U'))

bag_of_words = CountVectorizer(stop_words='english',
                               max_features=5000)
bag_of_words.fit(train_words)
train_words_binary_matrix = bag_of_words.transform(train_words)

tfidf = TfidfTransformer(smooth_idf=False)
tfidf.fit(train_words_binary_matrix)
train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

save_matrix(df_train, train_words_tfidf_matrix, train_output)
del df_train

df_test = get_df(test_input)
test_words = np.array(df_test.text.str.lower().values.astype('U'))
test_words_binary_matrix = bag_of_words.transform(test_words)
test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)

save_matrix(df_test, test_words_tfidf_matrix, test_output)
