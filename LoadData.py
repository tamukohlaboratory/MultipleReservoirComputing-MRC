import json
import os
import os.path
import numpy as np
import shutil
import csv


class SplittingData:

    def __init__(self, split_file, split, root, transforms=None, 
                 save_path_param=None, training_name_folder='DataTraining100',
                 validation_name_folder='DataValidation100',
                 test_name_folder='DataTesting100'):
        
        self.num_classes = self.get_num_class(split_file)
        self.split_file = split_file
        self.transforms = transforms
        self.root = root

        self.data = self.make_dataset(split_file, split, self.root, 
                                      save_path=save_path_param, 
                                      training_name_folder=training_name_folder,
                                      validation_name_folder=validation_name_folder, 
                                      test_name_folder=test_name_folder)

    def __len__(self):
        return len(self.data)

    def make_dataset(self, split_file, split, root, save_path=None,
                     training_name_folder='DataTraining100',
                     validation_name_folder='DataValidation100',
                     test_name_folder='DataTesting100'):
        
        print('make dataset for ', split)
        
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)

        ii = 0
        if (save_path is None):
            save_path = os.getcwd()
            print("save_path = ", save_path) 
        
        for vid in data.keys():
            # this code make the loop back to the main loop while does not meet the parameter such us: test, train, or val
            if split == 'train':
                if data[vid]['subset'] not in ['train', 'val']:
                    continue
            else:
                if data[vid]['subset'] != 'test':
                    continue
            #  take from the parameter which contain the dataset
            vid_root = os.path.join(root, 'WLASL2000')
            video_path = os.path.join(vid_root, vid + '.mp4')

            # split to dataset and write in folder
            tempLabel = str(data[vid]['action'][0])
            # array_class = np.array (['99','98','97','96','95','94','93','92','91','90'])
            array_class = np.array(['-1'])
            if ((tempLabel in array_class) is False):
                # get string label from action number
                tempLabel = self.get_class_from_list(tempLabel)
                
                if split == 'train':
                    new_dir = str(os.path.join(save_path, 
                                               training_name_folder, 
                                               tempLabel))
                    
                    if not os.path.isdir(new_dir):
                        os.makedirs(new_dir)
                        # print(new_dir)
                    shutil.copy(video_path, new_dir)
                elif split == 'val':
                    new_dir = str(os.path.join(save_path, 
                                               validation_name_folder, 
                                               tempLabel))
                    
                    if not os.path.isdir(new_dir):
                        os.makedirs(new_dir)
                        # print(new_dir)
                    shutil.copy(video_path, new_dir)
                elif split == 'test':
                    new_dir = str(os.path.join(save_path, 
                                               test_name_folder, 
                                               tempLabel))
                    
                    if not os.path.isdir(new_dir):
                        os.makedirs(new_dir)
                        # print(new_dir)
                    shutil.copy(video_path, new_dir)
            
            ii += 1
        # print("Skipped videos: ", count_skipping)
        # print(len(dataset))
        return dataset

    def get_num_class(self, split_file):
        classes = set()
        content = json.load(open(split_file))

        for vid in content.keys():
            class_id = content[vid]['action'][0]
            classes.add(class_id)

        return len(classes)

    def get_class_from_list(self, class_number):   
        with open(os.path.join(self.root, 'preprocess/wlasl_class_list.txt')) as wlasl_class_list:
            reader = csv.reader(wlasl_class_list, delimiter="\t")
            class_string = np.asarray(list(reader))
            index_array = np.where(class_string[:, 0] == str(class_number))
            if (len(index_array) > 0):
                return (f"{class_string[int(index_array[0]),1]}") 
            else:
                return ("")


if __name__ == "__main__":
    base_dir = "/home/bra1n/Documents/signLanguage"
    variableClass = '100'  # 2000
    train_split = os.path.join(base_dir, 'preprocess/nslt_{}.json'.format(variableClass))
    save_path_param = os.path.join(base_dir, 'paperNeuralComputing')
    # copying dataset into folder
    # next extract intokeypoint folder, firstly watch i3d method what is the effect on start and end of frame
    # had been executed
    dataset = SplittingData(train_split, 'train', base_dir, None, 
                            save_path_param=save_path_param,
                            training_name_folder='DataTraining{}'.format(variableClass),
                            validation_name_folder='DataValidation{}'.format(variableClass),
                            test_name_folder='DataTesting{}'.format(variableClass))
    dataset_test = SplittingData(train_split, 'test', base_dir, None, 
                                 save_path_param=save_path_param,
                                 training_name_folder='DataTraining{}'.format(variableClass),
                                 validation_name_folder='DataValidation{}'.format(variableClass),
                                 test_name_folder='DataTesting{}'.format(variableClass))
    dataset = SplittingData(train_split, 'val', base_dir, None, 
                            save_path_param=save_path_param,
                            training_name_folder='DataTraining{}'.format(variableClass),
                            validation_name_folder='DataValidation{}'.format(variableClass),
                            test_name_folder='DataTesting{}'.format(variableClass))