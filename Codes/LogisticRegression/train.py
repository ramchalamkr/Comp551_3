def train():
  import pandas
  from sklearn import svm
  import numpy
  from sklearn.externals import joblib
  import scipy.misc
  from sklearn.linear_model import LogisticRegression

  x = numpy.fromfile('train_x_clean.bin', dtype='uint8')
  x = x.reshape((100000, 3600)).astype('float64')

  df = pandas.read_csv('train_y.csv')
  y = df.as_matrix()
  a,b = y.shape

  y = y[:, b-1]

  #2-fold, fold=1
  clf = LogisticRegression(solver='sag', max_iter=500, C=1, n_jobs=2)
  clf.fit(x[0:70000], y[0:70000])

  joblib.dump(clf, "LR6060_train_C1_1.pkl", compress=3)

  score0 = numpy.zeros((2))
  score00 = numpy.zeros((2))
  score0[0] = clf.score(x[70000:100000], y[70000:100000]) #cross validation
  score00[0] = clf.score(x[0:70000], y[0:70000]) #training
  #fold = 2
  clf = LogisticRegression(solver='sag', max_iter=500, C=1, n_jobs=2)
  clf.fit(x[30000:100000], y[30000:100000])

  joblib.dump(clf, "LR6060_train_C1_2.pkl", compress=3)

  score0[1] = clf.score(x[0:30000], y[0:30000]) #cross validation
  score00[1] = clf.score(x[30000:100000], y[30000:100000]) #training
  numpy.savetxt("val_score_lr6060_c1.txt", score0)
  numpy.savetxt("train_score_lr6060_c1.txt", score00)

  clf = LogisticRegression(solver='sag', max_iter=500, C=1e-4, n_jobs=2)
  clf.fit(x[0:70000], y[0:70000])

  joblib.dump(clf, "LR6060_train_C1e_4_1.pkl", compress=3)

  score1 = numpy.zeros((2))
  score11 = numpy.zeros((2))
  score1[0] = clf.score(x[70000:100000], y[70000:100000])
  score11[0] = clf.score(x[0:70000], y[0:70000])

  clf = LogisticRegression(solver='sag', max_iter=500, C=1e-4, n_jobs=2)
  clf.fit(x[30000:100000], y[30000:100000])

  joblib.dump(clf, "LR6060_train_C1e_4_2.pkl", compress=3)

  score1[1] = clf.score(x[0:30000], y[0:30000])
  score11[1] = clf.score(x[30000:100000], y[30000:100000])

  numpy.savetxt("val_score_lr6060_c1e_4.txt", score1)
  numpy.savetxt("train_score_lr6060_c1e_4.txt", score1)

  if (score0[0] + score0[1] > score1[0] + score1[1]):
    inv_reg = 1
    name = "LR6060_train_C1_full.pkl"
  else:
    inv_reg = 1e-4
    name = "LR6060_train_C1e_4_full.pkl"

  clf = LogisticRegression(solver='sag', max_iter=500, C=inv_reg, n_jobs=2)
  clf.fit(x[0:100000], y[0:100000])

  joblib.dump(clf, name, compress=3)

  return name