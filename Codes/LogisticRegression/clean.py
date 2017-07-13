def clean():
  import numpy

  x = numpy.fromfile('test_x.bin', dtype='uint8')
  x = x.reshape((20000 * 3600))
  a, = x.shape
  y = numpy.zeros(a, dtype='uint8')

  for i in range(a):
    if (x[i] > 200):
	y[i] = x[i]

  y.tofile('test_x_clean.bin')
  x = 0
  y = 0

  x = numpy.fromfile('train_x.bin', dtype='uint8')
  x = x.reshape((100000 * 3600))
  a, = x.shape
  y = numpy.zeros(a, dtype='uint8')

  for i in range(a):
    if (x[i] > 200):
	y[i] = x[i]

  y.tofile('train_x_clean.bin')