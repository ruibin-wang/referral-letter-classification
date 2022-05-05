import json
import os
from function_sets import read_json, write_to_json
import pandas as pd


os.system("copy C:\Ruibin\Code\Preprocessing_referral_letter\data\symp_diagnosis_relation_training.csv C:\Ruibin\Code\BERT_text_classification\data")
os.system("copy C:\Ruibin\Code\Preprocessing_referral_letter\data\complaint_training_data.csv C:\Ruibin\Code\BERT_text_classification\data")


os.system("copy C:\Ruibin\Code\Preprocessing_referral_letter\data\\symp_diagnosis_relation_training.json C:\Ruibin\Code\BERT_text_classification\data")
os.system("copy C:\Ruibin\Code\Preprocessing_referral_letter\data\\filted_complaint_training_data.json C:\Ruibin\Code\BERT_text_classification\data")


symp_diagnosis_relation_training = read_json('./data/symp_diagnosis_relation_training.json')
filted_complaint_training_data = read_json('./data/filted_complaint_training_data.json')

combine_complaint_symp = []

for index in range(0, len(symp_diagnosis_relation_training)):
    templist = filted_complaint_training_data[index]['complaint'] + ' [SEP] ' + symp_diagnosis_relation_training[index]['symptoms']

    combine_complaint_symp.append(
        {'diagnosis': symp_diagnosis_relation_training[index]['diagnosis'], 'symptoms': templist}
    )

write_to_json('./data/combine_complaint_symp.json', combine_complaint_symp)
combine_complaint_symp = pd.read_json('./data/combine_complaint_symp.json')
combine_complaint_symp.head()
combine_complaint_symp.to_csv('./data/combine_complaint_symp.csv', index=False)









