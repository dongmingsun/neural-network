from builtins import range
from six.moves import cPickle as pickle
import numpy as np
from numpy import genfromtxt
import os
import platform
import re
from PIL import Image
import glob


def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    # print(filename)
    try:
        open(filename)
    except:
        print("fail to open file")

    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    # print(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000,
                     num_validation=1000,
                     num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '/vol/bitbucket/ds4317/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }


def load_FER2013_raw():
    fer_path = '/vol/bitbucket/ds4317/datasets/FER2013'
    # load labels
    # labels = genfromtxt(os.path.join(fer_path, 'labels_public.txt'), delimiter=',')
    # labels = labels[1:, 1]
    labels = np.load(os.path.join(fer_path, 'labels.npy'))

    train_num = len(glob.glob(os.path.join(fer_path, 'Train/*.jpg')))
    test_num = len(glob.glob(os.path.join(fer_path, 'Test/*.jpg')))
    print('Train sample number: ' + str(train_num))
    print('Test sample number: ' + str(test_num))
    # train dataset
    Xtr = np.zeros((train_num, 48, 48))
    Ytr = np.zeros((train_num, 1), dtype=int)
    # test dataset
    Xte = np.zeros((test_num, 48, 48))
    Yte = np.zeros((test_num, 1), dtype=int)

    i = 0
    for filename in glob.glob(os.path.join(fer_path, 'Train/*.jpg')):
        # read image and save image data into Xtr
        img = Image.open(filename).convert('L')
        Xtr[i, :, :] = np.asarray(img)
        # save corresponding label into Ytr
        img_num = int(re.findall('\d+', filename)[-1])
        Ytr[i] = labels[img_num - 1, 1]
        i += 1
        printProgressBar(
            i,
            train_num,
            prefix='Loading train...',
            suffix='Complete',
            length=50)

    i = 0
    for filename in glob.glob(os.path.join(fer_path, 'Test/*.jpg')):
        # read image and save image data into Xtr
        img = Image.open(filename).convert('L')
        Xte[i, :, :] = np.asarray(img)
        # save corresponding label into Ytr
        img_num = int(re.findall('\d+', filename)[-1])
        Yte[i] = labels[img_num - 1, 1]
        i += 1
        printProgressBar(
            i,
            test_num,
            prefix='Loading test...',
            suffix='Complete',
            length=50)

    print('data loaded...')

    np.save(os.path.join(fer_path, 'Xtr'), Xtr)
    np.save(os.path.join(fer_path, 'Xte'), Xte)
    np.save(os.path.join(fer_path, 'Ytr'), Ytr)
    np.save(os.path.join(fer_path, 'Yte'), Yte)

    return Xtr, Ytr, Xte, Yte
    # return Xtr, Ytr, Xte, Yte


def get_FER2013_data(num_training=28709 - 1000,
                     num_validation=1000,
                     num_test=3589,
                     subtract_mean=True):
    fer_path = '/vol/bitbucket/ds4317/datasets/FER2013'
    X_train = np.load(os.path.join(fer_path, 'Xtr.npy'))
    y_train = np.load(os.path.join(fer_path, 'Ytr.npy'))
    X_test = np.load(os.path.join(fer_path, 'Xte.npy'))
    y_test = np.load(os.path.join(fer_path, 'Yte.npy'))

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Package data into a dictionary
    return {
        'X_train': np.expand_dims(X_train, 1),
        'y_train': y_train.reshape(-1, ),
        'X_val': np.expand_dims(X_val, 1),
        'y_val': y_val.reshape(-1, ),
        'X_test': np.expand_dims(X_test, 1),
        'y_test': y_test.reshape(-1, ),
    }


def get_data_from_path(data_path):
    mean_image = np.load('./src/utils/mean_image.npy')
    mean_image = mean_image.reshape(1, 1, 48, 48)
    test_number = len(glob.glob(os.path.join(data_path, '*.jpg')))
    print('Test sample number: ' + str(test_number))
    X_test = np.zeros((test_number, 1, 48, 48))
    i = 0
    for filename in sorted(glob.glob(os.path.join(data_path, '*.jpg'))):
        # read image and save image data into Xtr
        img = Image.open(filename).convert('L')
        X_test[i, 0, :, :] = np.asarray(img)
        i += 1
        printProgressBar(
            i,
            test_number,
            prefix='Loading test data...',
            suffix='Complete',
            length=50)

    X_test -= mean_image
    print(X_test.shape)
    return X_test
