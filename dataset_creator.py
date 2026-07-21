import random
import csv 

output_data_train = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/training/training_set.csv"
output_data_test = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/testing/testing_set.csv"
output_data_validate = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/validation/validating_set.csv"
output_labels_train = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/training/training_labels.csv"
outpit_labels_test = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/testing/testing_labels.csv"
output_labels_validate = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/validation/validating_labels.csv"

iteration = 1 

while iteration <=3:
    if iteration == 1:
            output = output_data_train
            labels = output_labels_train
    elif iteration == 2:
            output = output_data_test
            labels = outpit_labels_test
    elif iteration == 3:
            output = output_data_validate
            labels = output_labels_validate

    with open(output, 'w', newline='') as datafile:
        fieldnames = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'] 
        writer = csv.DictWriter(datafile, fieldnames=fieldnames)
        writer.writeheader()

    with open(labels, 'w', newline='') as labelsfile:
        fieldnames = ['label']
        writer = csv.DictWriter(labelsfile, fieldnames = fieldnames)
        writer.writeheader()


    for i in range (0,999):
        x = []
        for j in range(0, 6):
            x.append(random.randint(0,1))
            c = 0
            label = 0
            k = 0
            for k in range(0, len(x)):
                if x[k] > 0: c+=1
            if c >= 0.5*len(x): label = 1

        data = {'x0':x[0], 'x1':x[1], 'x2':x[2], 'x3':x[3], 'x4':x[4], 'x5':x[5]}
        label = {'label': label}
        with open(output, 'a', newline='') as datafile:
            fieldnames = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'label'] 
            writer = csv.DictWriter(datafile, fieldnames=fieldnames)
            writer.writerows([data])
        with open(labels, 'a', newline='') as labelsfile:
            fieldnames = ['label']
            writer = csv.DictWriter(labelsfile, fieldnames=fieldnames)
            writer.writerows([label])
    iteration+=1



