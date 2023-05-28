# Through the files in csvs, the files in the txt directory are generated,
# and the generated files are java files that need to be converted into img

import numpy as np
import pandas as pd
import os


def extract_csv_instances(path):
    csv_instances = pd.read_csv(path)

    # get filename
    instances = np.array(csv_instances['file_name'])
    instances = instances.tolist()

    # get label
    labels = np.array(csv_instances['bug'])
    labels = labels.tolist()
    for index, label in enumerate(labels):
        if label > 1:
            labels[index] = 1

    return instances, labels


def parse_source(project_root_path, csv_file_instances, csv_file_labels, package_heads):
    count = 0
    existed_paths_and_labels = []
    for dir_path, dir_names, file_names in os.walk(project_root_path):

        # If there is no file under the folder, skip the folder directly
        if len(file_names) == 0:
            continue

        index = -1
        for _head in package_heads:
            index = int(dir_path.find(_head))
            if index >= 0:
                break
        if index < 0:
            continue

        package_name = dir_path[index:]
        package_name = package_name.replace(os.sep, '.')

        for file in file_names:
            if file.endswith('java'):
                file_name = os.path.splitext(file)[0]

                # Traverse the csv file, if there is one, save the file path and label
                for index, instance in enumerate(csv_file_instances):
                    if str(package_name + "." + str(file_name)) == instance:
                        file_real_path = "../data/"+dir_path+'/'+file
                        file_real_path = file_real_path.replace('\\', '/')
                        existed_paths_and_labels.append([file_real_path, csv_file_labels[index]])
                        count += 1
                        break

    for csv_file_instance in csv_file_instances:
        if csv_file_instance not in existed_paths_and_labels[0]:
            print('This file is not in csv list:' + csv_file_instance)

    print("data size : " + str(count))
    return existed_paths_and_labels


# Create a file under the txt folder, put the path, put the label
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com', 'fr']
root_path_csvs = '../data/csvs/'
root_path_archives = '../data/archives/'
root_path_txt = '../data/txt/'

for csv_file in os.listdir(root_path_csvs):
    # Get the project name from source project
    project_name = os.path.splitext(csv_file)[0]
    print(project_name + " begin!")

    # Get the tagged files in csv
    path_csv = root_path_csvs + project_name + '.csv'
    csv_file_instances, csv_file_labels = extract_csv_instances(path_csv)

    # get code folder
    path_archives = root_path_archives + project_name
    existed_paths_and_labels = parse_source(path_archives, csv_file_instances, csv_file_labels, package_heads)
    np.savetxt(root_path_txt+project_name+'.txt', existed_paths_and_labels, fmt="%s", delimiter=" ")

    print(project_name + " done!")