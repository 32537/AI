
#运行命令
# C:/Users/32537/.conda/envs/thyroid_ML/python.exe -m streamlit run c:/Users/32537/Desktop/MLtest/Untitled-1.py


from pycaret.classification import *
import streamlit as st
import numpy as np
import random


# 固定随机种子
np.random.seed(42)
random.seed(42)



#加载模型
loaded_model = load_model(r'Logistic Regression') #不需要后缀



#对采集数据进行字典匹配转换为01 后期进入模型


Halo_sign_dict = {'absent':0, 'Exists':1}
Gender_dict = {"Male":0,"Female":1}
Composition_dict = {'Others':0, 'Solid':1}
Shape_dict = {'Others':0, 'Microlobulated':1}
Echogenicity_dict = {'Others':0, 'Hypoechogenicity':1}
Echogenic_foci_dict = {'Others':0, 'Microcalcification':1}
Margin_dict = {'Smooth':0, 'Irregular':1}
ATR_dict = {'Wider_than_tall':0, 'Taller_than_wide':1}
Peri_BFS_dict = {'Absent':0, 'Less':1, 'Rich':2}

Pathological_diagnosis_dict = {0:'Benign', 1:'Malignant'}



#构建信息输入界面

#性别
Gender= st.selectbox(
    label = 'Please select your gender',
    options = ('Male', 'Female'),
    index = 1,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)

#年龄
Age = st.number_input(
    label = 'Please input your age(years):',
    # min_value = 15.4,
    # max_value = 83,
    value = 30,
    #step = 1,
    format = '%d',
    #help = 'Please input your age in years'
)


#Peri_BFS Absent/Less/Rich
Peri_BFS = st.selectbox(
    label = 'Please select your Peri_BFS',
    options = ('Absent', 'Less', 'Rich'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)

#Maximum_diameter
Maximum_diameter = st.number_input(
    label = 'Please input your Maximum_diameter(mm):',
    # min_value = 0,
    # max_value = 100,
    value = 0.5,
    step = 0.1,
    format = '%.1f',
    help = 'Please input your Maximum_diameter in mm'
)

#Composition Others/Solid
Composition = st.selectbox(
    label = 'Please select your Composition',
    options = ('Others', 'Solid'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)

#Shape  Microlobulated/Others
Shape = st.selectbox(
    label = 'Please select your Shape',
    options = ('Microlobulated', 'Others'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)

#Echogenicity Others/Hypoechogenicity
Echogenicity = st.selectbox(
    label = 'Please select your Echogenicity',
    options = ('Others', 'Hypoechogenicity'),
    index = 1,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)

#Echogenic_foci Others/Microcalcification
Echogenic_foci = st.selectbox(
    label = 'Please select your Echogenic_foci',
    options = ('Others', 'Microcalcification'),
    index = 1,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)

#Margin Smooth/Irregular
Margin = st.selectbox(
    label = 'Please select your Margin',
    options = ('Smooth', 'Irregular'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)

#ATR Wider_than_tall/Taller_than_wide
ATR = st.selectbox(
    label = 'Please select your ATR',
    options = ('Wider_than_tall', 'Taller_than_wide'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)




#对采集数据进行转换处理 生成predict_data数据框

import pandas as pd
predict_data = pd.DataFrame({
    'Age':[Age],
    'Gender':[Gender_dict[Gender]],
    'Maximum_Diameter':[Maximum_diameter],
    'Peri_BFS':[Peri_BFS_dict[Peri_BFS]],
    'Composition':[Composition_dict[Composition]],
    'Shape':[Shape_dict[Shape]],
    'Echogenicity':[Echogenicity_dict[Echogenicity]],
    'Echogenic_Foci':[Echogenic_foci_dict[Echogenic_foci]],
    'Margin':[Margin_dict[Margin]],
    'ATR':[ATR_dict[ATR]]
})




#预测
if st.button("Predict"):
    predition = predict_model(loaded_model, data=predict_data, raw_score = True,probability_threshold=0.4821)
    #predition
    #取出数据
    prediction_label = predition.iloc[0]['prediction_label']
    # prediction_label
    #显示数据框？

    if prediction_label == 0:
        prediction_score = predition.iloc[0]['prediction_score_0'] * 100
    elif prediction_label == 1:
        prediction_score = predition.iloc[0]['prediction_score_1'] * 100
    
    st.write(f"Predicted Class: {Pathological_diagnosis_dict[prediction_label]}")
    st.write(f"Prediction Probability: {prediction_score}%")

    #生成建议
    if prediction_label == 1:
        advice = (
            f"According to our model, you have a high risk of pathological diagnosis malignant. "
            f"The model predicts that your probability of having heart disease is {prediction_score:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "We recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    elif prediction_label == 0:
        advice = (
            f"According to our model, you have a low risk of pathological diagnosis malignant. "
            f"The model predicts that your probability of not having heart disease is {prediction_score:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)




