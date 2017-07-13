def predict(model_name):
  import pandas
  import numpy
  from sklearn.externals import joblib
  import scipy.misc
  from sklearn.linear_model import LogisticRegression

  clf = joblib.load(model_name);
  df = pandas.DataFrame(columns=('Id', 'Prediction'))

  x = numpy.fromfile('test_x_clean.bin', dtype='uint8')
  x = x.reshape((20000, 3600)).astype('float64')

  y = clf.predict(x)

  for i in range(20000):
    df.loc[i] = [i, int(y[i])]

  df.to_csv('final_prediction_logistic.csv', columns=('Id', 'Prediction'), index=False)