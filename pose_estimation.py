#!/usr/bin/python2
from __future__ import print_function
import progressbar
import transformations as trans
import numpy as np
import lmdb
import cv2
import caffe
import argparse
import subprocess
from matplotlib import pyplot as plt

MODEL_LOCATION = "/media/sebastien/Storage/"

RAD2DEGREE = 180.0 / np.pi

MAX_ITERS = {"Street": 2923, "StMarysChurch": 530, "StMarysChurchResNet": 530, "StMarysChurchSqueezeNet": 530, "KingsCollege": 343, "Soccer": 400}

CONVERSIONS = {"Street": lambda x: convertStreet(x),
                "StMarysChurch": lambda x: convertStMarys(x),
                "KingsCollege": lambda x: convertKings(x),
                "Soccer": lambda x: convertSoccer(x)}

STRIDE = 40

def convertDataset(dataset_location, dataset_file):
    dataset = readFile(dataset_location, dataset_file)
    # print(dataset)
    if "Soccer" in dataset_location:
        converted_dataset = convertSoccerLabels(dataset, dataset_location)
    else:
        converted_dataset = convertLabels(dataset, dataset_location)
    # print(label)

    return converted_dataset


def saveDataset(dataset, dataset_location, dataset_file):
    with open(dataset_location + dataset_file, "w") as file:
        for entry in dataset:
            file.write(" ".join([str(x) for x in entry]) + '\n')


def saveLMDB(dataset, dataset_location, dataset_file):
    np.random.shuffle(dataset)

    env = lmdb.open(data_location + dataset_file, map_size=int(1e12))

    progress = progressbar.ProgressBar(maxval=len(dataset))
    print("SAVING LMDB")
    progress.start()
    for i, entry in enumerate(dataset):
        image_location = entry[0]
        pose = tuple(entry[1:])

        X = cv2.imread(image_location)
        X = cv2.resize(X, (455,256))    # to reproduce PoseNet results, please resize the images so that the shortest side is 256 pixels
        X = np.transpose(X,(2,0,1))
        im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))
        im_dat.float_data.extend(pose)

        str_id = '{:0>10d}'.format(i)
        with env.begin(write=True) as txn:
            txn.put(str_id, im_dat.SerializeToString())
        # count = count+1

        progress.update(i+1)

    print("")

    env.close()


def readFile(dataset_location, dataset_file):
    dataset = []
    for line in open(dataset_location + dataset_file):
        string_elements = line[:-1].split(" ")
        float_elements = [string_elements[0]]
        float_elements.extend([float(x) for x in string_elements[1:]])
        dataset.append(float_elements)
    return dataset


def convertLocation(x, y, z):
    return [x, y]


def convertOrientation(w, p, q, r):
    euler = np.array(trans.euler_from_quaternion([w, p, q, r]))

    angle = euler[1]
    units = angleToUnitCircle(angle)

    return units


def angleToUnitCircle(angle):
    theta_x = np.cos(angle)
    theta_y = -np.sin(angle)

    return [theta_x, theta_y]


def unitCircleToAngle(theta_x, theta_y):
    angle = np.arctan2(theta_y, theta_x)

    return angle


def convertLabels(dataset, dataset_location):
    print("STREET SCENE")
    converted_dataset = []
    progress = progressbar.ProgressBar(maxval=len(dataset))
    print("CONVERTING LABELS")
    progress.start()

    for i, [image_location, x, z, y, w, p, q, r] in enumerate(dataset):
        converted_location = convertLocation(x, y, z)
        converted_orientation = convertOrientation(w, p, q, r)

        converted = [dataset_location + image_location]
        converted.extend(converted_location)
        converted.extend(converted_orientation)
        converted_dataset.append(converted)

        progress.update(i+1)
    print("")
    return converted_dataset


def convertImageLocation(image_location):
    image_id = image_location[len("Screenshots/Screenshot"):-4]
    new_image_location = "Screenshots/ScreenShot%05d.png" % int(image_id)
    return new_image_location


def convertSoccerLabels(dataset, dataset_location):
    print("SOCCER")
    converted_dataset = []
    progress = progressbar.ProgressBar(maxval=len(dataset))
    print("CONVERTING LABELS")
    progress.start()

    for i, [image_location, x, y, z, angle] in enumerate(dataset):
        theta = angle / RAD2DEGREE
        converted_location = convertLocation(x, y, z)
        converted_orientation = angleToUnitCircle(angle)

        image_location = convertImageLocation(image_location)

        converted = [dataset_location + image_location]
        converted.extend(converted_location)
        converted.extend(converted_orientation)
        converted_dataset.append(converted)

        progress.update(i+1)

    print("")
    return converted_dataset


def convertSoccer(dataset):
    half_dataset = []

    for i, [image_location, x, y, t_x, t_y] in enumerate(dataset):
        half_x = x
        half_y = y

        half_tx = t_x
        half_ty = t_y

        half = 0

        if y < 0:
            half_y = -y
            half_x = -x
            half_tx = -t_x
            half_ty = -t_y
            half = 1

        half_dataset.append([image_location, half_x, half_y, half_tx, half_ty, half])

    scale = 20
    xs = [x[1] for x in dataset]
    ys = [x[2] for x in dataset]
    txs = [x[3] for x in dataset]
    tys = [x[4] for x in dataset]

    xlim = [min(xs), max(xs)]
    ylim = [min(ys), max(ys)]
    subplot = plt.subplot(1,2,1)
    subplot.scatter(xs, ys, c="red")
    for i in range(len(txs)):
        # print(txs[i], tys[i])
        subplot.plot([xs[i], xs[i] - scale*tys[i]], [ys[i], ys[i] + scale*txs[i]], c="red")
    subplot.set_xlim(xlim)
    subplot.set_ylim(ylim)

    xs = [x[1] for x in half_dataset]
    ys = [x[2] for x in half_dataset]
    txs = [x[3] for x in half_dataset]
    tys = [x[4] for x in half_dataset]
    subplot = plt.subplot(1,2,2)
    subplot.scatter(xs, ys, c="blue")
    for i in range(len(txs)):
        subplot.plot([xs[i], xs[i] - scale*tys[i]], [ys[i], ys[i] + scale*txs[i]], c="blue")
    subplot.set_xlim(xlim)
    subplot.set_ylim(ylim)

    plt.show()
    return half_dataset


def average(l):
    s = 0
    for i in l:
        s += i
    return s / len(l)


def convertStreet(dataset):
    xs = [x[1] for x in dataset]
    ys = [x[2] for x in dataset]

    half_dataset = []

    progress = progressbar.ProgressBar(maxval=len(dataset))
    print("CREATING HALF DATASET")
    progress.start()
    for i, [image_location, x, y, t_x, t_y] in enumerate(dataset):
        avr = average(ys[max(0, i-STRIDE):min(len(xs), i + STRIDE)])

        x_half = x
        y_half = abs(y - avr)

        if y - avr < 0:
            t_x_half = t_x
            t_y_half = -t_y
            half = 1
        else:
            t_x_half = t_x
            t_y_half = t_y
            half = 0

        half_dataset.append([image_location, x_half, y_half, t_x_half, t_y_half, half])

        progress.update(i+1)
    print("")

    scale = 0.1
    xs = [x[1] for x in dataset]
    ys = [x[2] for x in dataset]
    txs = [x[3] for x in dataset]
    tys = [x[4] for x in dataset]

    xlim = [min(xs), max(xs)]
    ylim = [min(ys), max(ys)]
    subplot = plt.subplot(1,2,1)
    subplot.scatter(xs, ys, c="red")
    for i in range(len(txs)):
        # print(txs[i], tys[i])
        subplot.plot([xs[i], xs[i] - scale*tys[i]], [ys[i], ys[i] + scale*txs[i]], c="red")
    subplot.set_xlim(xlim)
    subplot.set_ylim(ylim)

    xs = [x[1] for x in half_dataset]
    ys = [x[2] for x in half_dataset]
    txs = [x[3] for x in half_dataset]
    tys = [x[4] for x in half_dataset]
    subplot = plt.subplot(1,2,2)
    subplot.scatter(xs, ys, c="blue")
    for i in range(len(txs)):
        subplot.plot([xs[i], xs[i] - scale*tys[i]], [ys[i], ys[i] + scale*txs[i]], c="blue")
    subplot.set_xlim(xlim)
    subplot.set_ylim(ylim)

    plt.show()

    return half_dataset


def convertStMarys(dataset):
    half_dataset = []

    for i, [image_location, x, y, t_x, t_y] in enumerate(dataset):
        half_x = x
        half_y = y

        half_tx = t_x
        half_ty = t_y

        half = 0

        if x < 15:
            half_x = -(x-15) + 15
            half_ty = -t_y
            half = 1

        half_dataset.append([image_location, half_x, half_y, half_tx, half_ty, half])

    scale = 5
    xs = [x[1] for x in dataset]
    ys = [x[2] for x in dataset]
    txs = [x[3] for x in dataset]
    tys = [x[4] for x in dataset]

    xlim = [min(xs), max(xs)]
    ylim = [min(ys), max(ys)]
    subplot = plt.subplot(1,2,1)
    subplot.scatter(xs, ys, c="red")
    for i in range(len(txs)):
        # print(txs[i], tys[i])
        subplot.plot([xs[i], xs[i] - scale*tys[i]], [ys[i], ys[i] + scale*txs[i]], c="red")
    subplot.set_xlim(xlim)
    subplot.set_ylim(ylim)

    xs = [x[1] for x in half_dataset]
    ys = [x[2] for x in half_dataset]
    txs = [x[3] for x in half_dataset]
    tys = [x[4] for x in half_dataset]
    subplot = plt.subplot(1,2,2)
    subplot.scatter(xs, ys, c="blue")
    for i in range(len(txs)):
        subplot.plot([xs[i], xs[i] - scale*tys[i]], [ys[i], ys[i] + scale*txs[i]], c="blue")
    subplot.set_xlim(xlim)
    subplot.set_ylim(ylim)

    plt.show()
    return half_dataset


def convertKings(dataset):
    half_dataset = []
    print("CREATING HALF DATASET")
    progress = progressbar.ProgressBar(maxval=len(dataset))
    progress.start()
    for i, [image_location, x, y, t_x, t_y] in enumerate(dataset):

        half_x = x
        half_y = y
        half_tx = t_x
        half_ty = t_y
        half = 0

        if half_x < 0:
            half_x = -half_x
            half_ty = - t_y
            half = 1

        half_dataset.append([image_location, half_x, half_y, half_tx, half_ty, half])

        progress.update(i+1)
    print("")

    scale = 1
    xs = [x[1] for x in dataset]
    ys = [x[2] for x in dataset]
    txs = [x[3] for x in dataset]
    tys = [x[4] for x in dataset]
    plt.subplot(1,2,1)
    plt.scatter(xs, ys, c="red")
    for i in range(len(txs)):
        plt.plot([xs[i], xs[i] - scale*tys[i]], [ys[i], ys[i] + scale*txs[i]], c="red")

    xs = [x[1] for x in half_dataset]
    ys = [x[2] for x in half_dataset]
    txs = [x[3] for x in half_dataset]
    tys = [x[4] for x in half_dataset]
    plt.subplot(1,2,2)
    plt.scatter(xs, ys, c="blue")
    for i in range(len(txs)):
        plt.plot([xs[i], xs[i] - scale*tys[i]], [ys[i], ys[i] + scale*txs[i]], c="blue")

    plt.show()
    return half_dataset

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        print(" ".join(cmd))
        raise subprocess.CalledProcessError(return_code, cmd)

def angleError(prediction, label):
    diff = np.abs(prediction - label) % (2*np.pi)
    if diff > np.pi:
        return 2*np.pi - diff
    return diff

def testModel(data_location, model_file, model_state, max_iter):
    caffe.set_mode_gpu()
    net = caffe.Net(data_location + model_file, data_location + "snaps/" + model_state, caffe.TEST)

    xy_errors = []
    or_errors = []
    half_errors = []

    predictions = []
    labels = []
    half_labels = []

    for i in range(max_iter):
        net.forward()

        predicted_xy = np.squeeze(net.blobs['cls3_fc_xy'].data)
        predicted_or = np.squeeze(net.blobs['cls3_fc_or'].data)
        predicted_or = predicted_or / np.linalg.norm(predicted_or)
        predicted_angle = unitCircleToAngle(predicted_or[0], predicted_or[1])

        labeled_xy = np.squeeze(net.blobs['label_xy'].data)
        labeled_or = np.squeeze(net.blobs['label_or'].data)
        labeled_angle = unitCircleToAngle(labeled_or[0], labeled_or[1])

        xy_error = np.linalg.norm(predicted_xy - labeled_xy)
        or_error = angleError(predicted_angle, labeled_angle) * RAD2DEGREE

        xy_errors.append(xy_error)
        or_errors.append(or_error)

        predictions.append(predicted_xy.copy())
        labels.append(labeled_xy.copy())

        if "label_half" in net.blobs:
            labeled_half = np.squeeze(net.blobs['label_half'].data)
            predicted_half = np.squeeze(net.blobs['cls3_fc_half_sig'].data)

            half_labels.append(labeled_half.copy())

            half_error = np.abs(labeled_half - predicted_half)
            half_errors.append(half_error)
            print ("%6d XY/OR/HALF: %7.2f   %7.2f   %7.2f" % (i, xy_error, or_error, half_error))
        else:
            print ("%6d XY/OR: %7.2f   %7.2f" % (i, xy_error, or_error))

    result_name = data_location + "result"
    if "label_half" in net.blobs:
        result_name += "_half"
    result_name += ".txt"

    result = np.concatenate((np.matrix(xy_errors), np.matrix(or_errors)), 0)

    if "label_half" in net.blobs:
        result = np.concatenate((result, np.matrix(half_errors)), 0)

    print("result shape", result.shape)
    print(result_name)

    np.savetxt(result_name, result.transpose(), "%5.3f")

    xy_average = float(sum(xy_errors))/len(xy_errors)
    or_average = float(sum(or_errors))/len(or_errors)

    if "label_half" in net.blobs:
        colors = [[0, 1, 0] if x < 0.5 else [1, 0, 0] for x in half_errors]
        plt.scatter(xy_errors, or_errors, c = colors)

        half_average = float(sum(half_errors))/len(half_errors)
        half_errors = sorted(half_errors)
        half_median = half_errors[len(half_errors)/2]

        xy_errors = sorted(xy_errors)
        or_errors = sorted(or_errors)

        xy_median = xy_errors[len(xy_errors)/2]
        or_median = or_errors[len(or_errors)/2]

        print ("Std XY/OR: %7.2f   %7.2f" % (np.std(xy_errors), np.std(or_errors)))

        print ("Avr    XY/OR/HALF: %7.2f   %7.2f   %7.2f" % (xy_average, or_average, half_average))
        print ("Median XY/OR/HALF: %7.2f   %7.2f   %7.2f" % (xy_median, or_median, half_median))

    else:
        plt.scatter(xy_errors, or_errors)

        xy_errors = sorted(xy_errors)
        or_errors = sorted(or_errors)

        xy_median = xy_errors[len(xy_errors)/2]
        or_median = or_errors[len(or_errors)/2]
        print ("Std XY/OR: %7.2f   %7.2f" % (np.std(xy_errors), np.std(or_errors)))

        print ("Avr    XY/OR: %7.2f   %7.2f" % (xy_average, or_average))
        print ("Median XY/OR: %7.2f   %7.2f" % (xy_median, or_median))

    xs = [x[0] for x in predictions]
    xls = [x[0] for x in labels]
    ys = [x[1] for x in predictions]
    yls = [x[1] for x in labels]
    print("x min", np.min(xs), np.min(xls))
    print("y min", np.min(ys), np.min(yls))

    print("x max", np.max(xs), np.max(xls))
    print("y max", np.max(ys), np.max(yls))

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, help="[convert|train|test]")
    parser.add_argument("dataset", type=str, help="any folder in clean")
    parser.add_argument("--half", help="if half version", action='store_true', dest="half", default=False)
    args = parser.parse_args()

    if not args.dataset in MAX_ITERS:
        print("Dataset must be one of the folders in the clean dir")

    data_location = MODEL_LOCATION + args.dataset + "/"

    if args.action == "convert":
        train_dataset = convertDataset(data_location, "dataset_train.txt")
        train_half_dataset = CONVERSIONS[args.dataset](train_dataset)
        saveDataset(train_dataset, data_location, "train_full.txt")
        saveDataset(train_half_dataset, data_location, "train_half.txt")
        saveLMDB(train_dataset, data_location, "train_full_lmdb")
        saveLMDB(train_half_dataset, data_location, "train_half_lmdb")

        test_dataset = convertDataset(data_location, "dataset_test.txt")
        test_half_dataset = CONVERSIONS[args.dataset](test_dataset)
        saveDataset(test_dataset, data_location, "test_full.txt")
        saveDataset(test_half_dataset, data_location, "test_half.txt")
        saveLMDB(test_dataset, data_location, "test_full_lmdb")
        saveLMDB(test_half_dataset, data_location, "test_half_lmdb")

    elif args.action == "train":
        pargs = ["caffe"]
        pargs += ["train"]
        pargs += ["--solver"]
        if args.half:
            pargs += [data_location + "model_half_solver.prototxt"]
        else:
            pargs += [data_location + "model_solver.prototxt"]

        pargs += ["--weights"]
        if "ResNet" in args.dataset:
            print("USING ALEXNET")
            pargs += [data_location + "../bvlc_alexnet.caffemodel"]
        elif "SqueezeNet" in args.dataset:
            pargs += [data_location + "../squeezenet.caffemodel"]
        else:
            pargs += [data_location + "../imagenet_googlenet.caffemodel"]

        print("Running: ", " ".join(pargs))
        for output in execute(pargs):
            print(output, end="")

    elif args.action == "test":
        if args.half:
            model_file = "model_half.prototxt"
            model_state = "model_half_iter_12000.caffemodel"
        else:
            model_file = "model.prototxt"
            model_state = "model_iter_16000.caffemodel"

        max_iter = MAX_ITERS[args.dataset]

        print(model_state)
        testModel(data_location, model_file, model_state, max_iter)
