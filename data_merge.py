from sklearn.utils import shuffle
from pandas import get_dummies
# from random import shuffle
import openpyxl
import numpy as np
import sklearn.preprocessing as labeler

class MRI_chosun_data():
    def __init__(self):
        self.class_array = []
        self.nn_data = []
        self.nn_label = []
        self.diag_type = "None" # or "clinic" or "new" or "PET"

        self.excel_option = ['P', 'T', 'V', 'merge']
        self.opt_dict_clinic = {
            'AD': 0,
            'CN': 1,
            'aMCI': 2,
            'naMCI': 3
        }

        self.opt_dict_new = {
            'aAD': 0,
            'NC': 1,
            'ADD': 2,
            'mAD': 3
        }

        self.class_option_dict_clinic = {
            'CN vs AD': ['CN', 'AD'],
            'CN vs MCI': ['CN', 'MCI'],
            'MCI vs AD': ['MCI', 'AD'],
            'CN vs MCI vs AD': ['CN', 'MCI', 'AD']
        }

        self.class_option_dict_new = {
            'NC vs ADD': ['NC', 'ADD'],
            'NC vs mAD': ['NC', 'mAD'],
            'NC vs aAD': ['NC', 'aAD'],
            'mAD vs ADD': ['mAD', 'ADD'],
            'aAD vs ADD': ['aAD', 'ADD'],
            'NC vs mAD vs ADD': ['NC', 'mAD', 'ADD'],
            'NC vs mAD vs aAD vs ADD': ['NC', 'mAD', 'aAD', 'ADD']
        }

        self.class_option_dict_pet = {
            'PET pos vs neg': ['positive', 'negative']
        }
#%%

    def set_diagnosis_type(self, type):
        self.diag_type = type
#%%

    def read_excel_data(self, excel_path):
        xl_file_name = excel_path
        xl_password = '!adai2018@#'
        xl = openpyxl.load_workbook(xl_file_name, read_only=True)
        ws = xl['Sheet1']
        self.data_excel = []
        for row in ws.rows:
            line = []
            for cell in row:
                line.append(cell.value)
            self.data_excel.append(line)
        # self.data_excel = np.array(self.data_excel)
        return self.data_excel

    def squeeze_excel(self, excel_option):
        '''
        because there are 3 lines for each patient basically,
        we have to choose how to handle it.

        choose only one line or merge all of them
        and then remove only zero column
        '''
        print('squeeze the excel.')
        if not excel_option in self.excel_option:
            print('the excel option in not proper.')
            print(excel_option, self.excel_option)
            assert False

        option_index = self.excel_option.index(excel_option)
        print('excel option : ',excel_option, option_index)
        '''
        ['MRI_id', 'gender', 'age', 'education', 'amyloid PET result', 
        'Clinic Diagnosis', 'New Diag', 'mtype', 'c4', ...]
        '''
        for i in range(1,len(self.data_excel)):
            '''
            should test if or not demo score help the model classify correctly.
            the age factor seems to provide useful information.
            but not sure about gender and education
            '''
            # print(label_info)
            gender = [self.convert_gender_to_int(self.data_excel[i][1])]
            # demo_score = gender + self.data_excel[i][2:4] # all factors
            # demo_score = self.data_excel[i][2:4] # age + edu
            demo_score = self.data_excel[i][2:3] # only age
            # demo_score = self.data_excel[i][3:4] # only education

            if (i-1)%3 == option_index: # if choose only one row : P V T
                line = self.data_excel[i][8:]
                label_info = self.data_excel[i][4:7]
                new_line = demo_score + line
                self.nn_data.append(new_line)
                self.nn_label.append(label_info)
                # print(len(self.data_excel[i]), len(line))
                # print(new_line)

            if option_index == 3 and i%3 == 1: # if use all three rows : merge
                line = [self.data_excel[i+k][8:] for k in range(3)]
                label_info = self.data_excel[i][4:7]
                new_line = demo_score + line[0] + line[1] + line[2]
                self.nn_data.append(new_line)
                self.nn_label.append(label_info)
                # print(len(self.data_excel[i][:10]), len(line[0]), len(line[1]), len(line[2]))
                # print(new_line)
        return self.nn_data, self.nn_label

    def remove_zero_column(self):
        self.nn_data = np.array(self.nn_data)
        l1, l2 = len(self.nn_data), len(self.nn_data[0])
        delete_col_count = 0
        print('remove zero value only columns.')
        print('matrix size : ',l1, 'X', l2)
        for col in range(l2):
            is_zero_col = True
            col_index = l2 - col - 1
            for row in range(l1):
                # print(type(self.nn_data[row][4]))
                if self.nn_data[row][col_index]:
                    # print(self.nn_data[row][4])
                    # print(self.nn_data[row][col_index])
                    is_zero_col = False
                    break

            if is_zero_col:
                # print('delete column.')
                delete_col_count += 1
                self.nn_data = np.delete(self.nn_data, col_index, 1)
            # assert False
        print('removed {} columns.\t{}=>{}'.format(delete_col_count, l2, len(self.nn_data[0])))
        print('' .format() )
        return self.nn_data, self.nn_label

#%%
    def convert_gender_to_int(self, gender:str)->int:
        if gender == 'M':   return 0
        elif gender == 'F': return 1
        else:
            print('inappropriate gender is entered : ', gender)
            assert False

    def label_pet(self, label, class_array):
        for i, c in enumerate(self.class_array):
            if c in label:
                return i
                # print(c, label)
                break
        print('there is no appropriate label name. :')
        print(self.class_array, label)
        assert False

    def label_clinic(self, label, class_array):
        for i, c in enumerate(self.class_array):
            if c in label:
                '''
                print(label, i) # check the labeling state                
                '''
                return i
        if 'MCI' in label or 'AD' in label or 'CN' in label:
            return -1
        print('there is no appropriate label name. :')
        print(self.class_array, label)
        assert False

    def label_new(self, label, class_array):
        for i, c in enumerate(self.class_array):
            if c in label:
                return i
        if 'NC' in label or 'AD' in label:
            return -1
        print('there is no appropriate label name. :')
        print(self.class_array, label)
        assert False
#%%
    def set_label_func(self, label_info, class_option):
        if self.diag_type == 'PET':
            self.class_array = self.class_option_dict_pet[class_option]
            self.label_name = label_info[:,0]
            # print(self.label_name, self.class_array)
            self.label_func = self.label_pet

        elif self.diag_type == 'clinic':
            self.class_array = self.class_option_dict_clinic[class_option]
            self.label_name = label_info[:,1]
            self.label_func = self.label_clinic

        elif self.diag_type == 'new':
            self.class_array = self.class_option_dict_new[class_option]
            self.label_name = label_info[:,2]
            self.label_func = self.label_new
        else:
            print('diagnosis type is wrong. : ', self.diag_type)
            assert False
        print('class option : {} / class array : {}'.format(class_option, self.class_array))

    def remove_minus_one_label(self):
        '''
            remove the -1 line of label and data
        '''
        print('remove the -1 line of label and data.')
        self.label_list = np.array(self.label_list)
        label_length = len(self.label_list)
        for row in range(len(self.label_list)):
            row_index = label_length - row - 1
            label = self.label_list[row_index]
            # print(row, len(self.label_list), row_index)
            if label == -1:
                self.data = np.delete(self.data, row_index, 0)
                self.label_list = np.delete(self.label_list, row_index, 0)

    def define_label_nn(self, label_info, class_option):
        '''
        :param label_info: it has 3 columns : pet, new, clinic
        :param class_option:  'PET pos vs neg', 'NC vs MCI vs AD' 'NC vs mAD vs aAD vs ADD'
        :return:

        when we use the class option as NC vs AD, we need to remove MCI line.
        '''
        is_print = False
        if is_print: pass
        print('start labeling...' )
        label_info = np.array(label_info)
        self.data = self.nn_data
        self.label_list = []
        # self.class_array = self.get_class_array(class_option)
        self.set_label_func(label_info, class_option)
        for i, label in enumerate(self.label_name):
            self.label_list.append(self.label_func(label, self.class_array))

        self.remove_minus_one_label()
        if is_print:
            print(len(self.data),len(self.label_list))
            print(type(self.data),type(self.label_list))
            # print(self.nn_data[0])
            print(self.label_list)

        return self.data, self.label_list

    def shuffle_data(self, data, label):
        print('shuffle the data and label - data length : ',len(data),' = ', len(label) )
        assert len(data)==len(label)
        random_list = [i for i in range(len(data))]
        '''
        choose between shuffle and shuffle_static
        # random_list = shuffle_static( ... )
        '''
        shuffle(random_list)
        # print(random_list)
        self.shuffle_data, self.shuffle_label = [],[]
        for index in random_list:
            self.shuffle_data.append(data[index])
            self.shuffle_label.append(label[index])

        assert len(label) == len(self.shuffle_label)
        return self.shuffle_data, self.shuffle_label

    def split_data_by_fold(self, data, label, fold_num):
        '''
        :return: return all possible train and test set according to the fold number.
        '''
        label_set = list(set(label))
        print('split the data into train and test by fold number. fold number : {} label set :{}' \
              .format(fold_num, label_set))
        separate_data = [[] for _ in range(len(label_set))]
        separate_label = [[] for _ in range(len(label_set))]
        # separate the data into different label list
        for i, l in enumerate(label):
            # print(i,l,label_count)
            separate_data[l].append(data[i])
            separate_label[l].append(label[i])
        # print(separate_label)
        label_count = [len(i) for i in separate_label]
        test_count = [count//fold_num for count in label_count]
        print(label_count, test_count)
        smaller_data_num = min(label_count)
        test_num = smaller_data_num//fold_num

        whole_set = []
        for fold_index in range(fold_num):
            train_data, train_label, test_data, test_label = [], [], [], []
            for i, one_label in enumerate(separate_data):
                if fold_index == fold_num - 1:
                    train_data = train_data + one_label[:test_count[i] * fold_index]
                    train_label = train_label + separate_label[i][:test_count[i] * fold_index]
                    test_data = test_data + one_label[test_count[i] * fold_index:]
                    test_label = test_label + separate_label[i][test_count[i] * fold_index:]
                    pass
                else:
                    train_data = train_data + one_label[:test_count[i] * fold_index] + one_label[test_count[i] * (fold_index + 1):]
                    train_label = train_label + separate_label[i][:test_count[i] * fold_index] + separate_label[i][test_count[i] * (fold_index + 1):]
                    test_data = test_data + one_label[test_count[i] * fold_index:test_count[i] * (fold_index + 1)]
                    test_label = test_label + separate_label[i][test_count[i] * fold_index:test_count[i] * (fold_index + 1)]
            # print(train_label)
            # print(test_label)
            print(len(train_label)+len(test_label), len(train_label), len(test_label))
            train_data = np.array(train_data)
            test_data = np.array(test_data)
            train_label = np.array(train_label)
            test_label = np.array(test_label)
            whole_set.append([train_data, train_label, test_data, test_label])
        return whole_set
#%%
def one_hot_pd(label):
    return get_dummies(label)

def shuffle_static(arr1, arr2):
    return shuffle(arr1, arr2, random_state=0)

def valence_class(data, label, class_num):
    print('Valence the number of train and test dataset')
    length = len(data)
    label_count = [0 for i in range(class_num)]
    label_count_new = [0 for i in range(class_num)]

    for i in sorted(label):
        label_count[i] += 1

    # print('label count : ', label_count)
    min_count = min(label_count)
    print('minimun count : ',min_count)
    new_data = []
    new_label = []
    for i, k in enumerate(label):
        if label_count_new[k] < min_count:
            new_data.append(data[i])
            new_label.append(label[i])
            label_count_new[k] += 1
    # print('new label count : ', label_count_new)
    print('down sampling : {} -> {}.'.format(label_count, label_count_new))
    return np.array(new_data), np.array(new_label)

# @datetime_decorator
def test_something_2():
    loader = MRI_chosun_data()
    loader.set_diagnosis_type('new')
    base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    loader.read_excel_data(excel_path)
    path_list = loader.extr_input_path_list(base_folder_path)
    print(path_list[0])
    loader.get_label_info_excel()
    loader.merge_info()

def normalize_col(X_, axis=0):
    assert len(np.amax(X_,axis=axis)) == len(X_[0])
    assert np.all(np.amax(X_,axis=axis) != 0)
    return (X_-np.amin(X_,axis=axis))/np.amax(X_, axis=axis)

def column(matrix, i, num):
    col = [row[i:i+num] for row in matrix]
    for row in matrix:
        if row[i:i+num] == []:
            print('Here is the criminal')
            print(row)
            assert False
    return col

def NN_dataloader(diag_type, class_option, excel_path, excel_option, fold_num):
    '''
    1. read excel data (O)
    2. squeeze 3 lines into 1 lines according to the options P V T merge (O)
    5. normalization -> should do this at first. separately according to the column.
    no. we don't have to do this at first. because column is preserved.
    3. remove zero value only column (O)
    3. make label list (O)
    4. shuffle (O)
    6. split train and test dataset (O)
    :return: train and test data and lable
    '''
    # "clinic" or "new" or "PET"
    # 'PET pos vs neg', 'NC vs MCI vs AD' 'NC vs mAD vs aAD vs ADD'
    # diag_type = "PET"
    # class_option = 'PET pos vs neg'
    # diag_type = "new"
    # class_option = 'NC vs mAD vs aAD vs ADD'
    # diag_type = "clinic"
    # class_option = 'NC vs AD' #'NC vs MCI vs AD'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    # excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    # excel_option = 'merge' # P V T merge
    loader = MRI_chosun_data()
    loader.set_diagnosis_type(diag_type)
    loader.read_excel_data(excel_path)
    loader.squeeze_excel(excel_option=excel_option)
    data, label_info = loader.remove_zero_column()
    data, label = loader.define_label_nn(label_info, class_option)
    print(label)
    # binarizer =
    # print()
    # assert False
    data = normalize_col(data)
    '''
    in this setting, we use static shuffle only
    '''
    # shuffle_data, shuffle_label = loader.shuffle_data(data, label)
    shuffle_data, shuffle_label = shuffle_static(data, label)
    return loader.split_data_by_fold(shuffle_data, shuffle_label, fold_num)

from imblearn.over_sampling import *
from imblearn.combine import *
def over_sampling(X_imb, Y_imb, sampling_option):
    print('starts over sampling ...', sampling_option)
    is_reshape = False
    if np.ndim(X_imb) == 1:
        is_reshape = True
        X_imb = X_imb.reshape(-1,1)
        print(X_imb.shape)

    if sampling_option == 'ADASYN':
        X_samp, Y_samp = ADASYN().fit_sample(X_imb, Y_imb)
    elif sampling_option == 'SMOTE':
        X_samp, Y_samp = SMOTE(random_state=4).fit_sample(X_imb, Y_imb)
    elif sampling_option == 'SMOTEENN':
        X_samp, Y_samp = SMOTEENN(random_state=0).fit_sample(X_imb, Y_imb)
    elif sampling_option == 'SMOTETomek':
        X_samp, Y_samp = SMOTETomek(random_state=4).fit_sample(X_imb, Y_imb)
    elif sampling_option == 'RANDOM':
        X_samp, Y_samp = RandomOverSampler(random_state=0).fit_sample(X_imb, Y_imb)
    elif sampling_option == 'BolderlineSMOTE':
        X_samp, Y_samp = BorderlineSMOTE().fit_sample(X_imb, Y_imb)
    elif sampling_option == 'None':
        X_samp, Y_samp = X_imb, Y_imb
    else:
        print('sampling option is not proper.', sampling_option)
        assert False

    if is_reshape:
        X_samp = np.squeeze(X_samp)
        print(X_samp.shape)

    imbalance_num = len(Y_imb)
    balance_num = len(Y_samp)
    print('over sampling from {:5} -> {:5}.'.format(imbalance_num, balance_num))
    return X_samp, Y_samp

if __name__ == '__main__':
    base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'# desktop setting
    # base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/test_sample_2'
    excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    # "clinic" or "new" or "PET"
    # 'PET pos vs neg', 'NC vs MCI vs AD' 'NC vs mAD vs aAD vs ADD'
    # diag_type = "PET"
    # class_option = 'PET pos vs neg'
    diag_type = "clinic"
    class_option = 'CN vs AD'  # 'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    # diag_type = "clinic"
    # class_option = 'MCI vs AD'#'MCI vs AD'#'CN vs MCI'#'CN vs AD' #'CN vs MCI vs AD'
    excel_option = 'merge'
    fold_num = 5
    # test_num = 10
    # is_split_by_num = False
    whole_set = NN_dataloader(diag_type, class_option, excel_path, excel_option, fold_num)
    print(np.shape(whole_set[0][0]))
    print(whole_set[0][0][:5])

'''
<<numpy array column api>> 
self.data_excel[:,0]

# arr1 = [6,5,4,3,2,1]
# arr2 = [1,2,3,4,5,6]
# for _ in range(10):
#     arr1_1, arr2_1 = shuffle_static(arr1,arr2)
#     print(arr1_1)
#     print(arr2_1)
# assert False
'''