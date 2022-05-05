import os

import numpy as np
import pandas as pd

print('#########################################')
print('Precessing....')


saved_path = '.\data\processed_training_data'

file_name_set = ["symp_diagnosis_relation_training", "complaint_training_data", "combine_complaint_symp"]


def replace_dot(df_train):
    temp_list = [idx.split('[SEP]') for idx in df_train['symptoms'].values]
    for idx in range(len(temp_list)):
        temp_list[idx][1] = temp_list[idx][1].replace('.', '')

    df_train['symptoms'] = np.array([idx[0] + ' [SEP] ' + idx[1] for idx in temp_list])

    return df_train



for file_idx in range(len(file_name_set)):

    text_path = '.\data\\' + file_name_set[file_idx] + '.csv'
    data_path = os.path.join(saved_path, file_name_set[file_idx])



    df_train_path = os.path.join(data_path, 'train.csv')
    df_val_path = os.path.join(data_path, 'val.csv')
    df_test_path = os.path.join(data_path, 'test.csv')



    df_train = pd.read_csv(df_train_path)
    df_val = pd.read_csv(df_val_path)
    df_test = pd.read_csv(df_test_path)

    original_text = pd.read_csv(text_path)


    if file_idx is 0:
        df_train['symptoms'] = np.array([idx.replace('.', '') for idx in df_train['symptoms'].values])
        df_val['symptoms'] = np.array([idx.replace('.', '') for idx in df_val['symptoms'].values])
        df_test['symptoms'] = np.array([idx.replace('.', '') for idx in df_test['symptoms'].values])
        original_text['symptoms'] = np.array([idx.replace('.', '') for idx in original_text['symptoms'].values])


    elif file_idx is 2:
        df_train = replace_dot(df_train)
        df_val = replace_dot(df_val)
        df_test = replace_dot(df_test)
        original_text = replace_dot(original_text)


    df_train.to_csv(os.path.join(data_path, 'no_dot_train.csv'), index=False, encoding="utf-8")
    df_val.to_csv(os.path.join(data_path, 'no_dot_val.csv'), index=False, encoding="utf-8")
    df_test.to_csv(os.path.join(data_path, 'no_dot_test.csv'), index=False, encoding="utf-8")
    original_text.to_csv(os.path.join(data_path, 'no_dot_original_text.csv'), index=False, encoding="utf-8")


print('Has already delete the dot!!')





