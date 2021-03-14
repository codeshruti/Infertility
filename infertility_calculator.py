
# coding: utf-8

# The dataset has been created using 9 input features:
# First_period_age:0 or 1
# Marriage_age:0 or 1 
# Irregular_menstrual_cycle:0 or 1
# Family_history:0 or 1
# Diabetes : 0 or 1
# Hypertension: 
# 0 – No
# 1- Yes
# Thyroid : 0 or 1
# PCOD: 0 or 1
# Genital infection :0 or 1
# 
# 
# 203 attributes are there

# In[11]:


from pandas import read_csv
import pandas as pd
filename = 'infertility.csv'
inf = read_csv(filename)
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier


# In[12]:


inf.head()


# In[13]:


inf.info()


# In[14]:


inf['Infertility'].value_counts()


# In[15]:


# Assigning categorical variables
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
inf['Infertility']= label_encoder.fit_transform(inf['Infertility']) 
inf['Diabetes'] = inf['Diabetes'].astype('category')
inf['Hypertension'] = inf['Hypertension'].astype('category')
inf['Thyroid'] = inf['Thyroid'].astype('category')
inf['Polycystic Ovary'] = inf['Polycystic Ovary'].astype('category')
inf['Genital Infection'] = inf['Genital Infection'].astype('category')
inf['First_period_age'] = inf['First_period_age'].astype('category')
inf['Marriage_age'] = inf['Marriage_age'].astype('category')
inf['Irregular_menstrual_cycle'] = inf['Irregular_menstrual_cycle'].astype('category')
inf['Family_history'] = inf['Family_history'].astype('category')


# In[16]:


inf.head()


# In[17]:


inf['Diabetes'] = inf['Diabetes'].cat.codes
inf['Hypertension'] = inf['Hypertension'].cat.codes
inf['Thyroid'] = inf['Thyroid'].cat.codes
inf['Polycystic Ovary'] = inf['Polycystic Ovary'].cat.codes
inf['Genital Infection'] = inf['Genital Infection'].cat.codes
inf['First_period_age'] = inf['First_period_age'].cat.codes
inf['Marriage_age'] = inf['Marriage_age'].cat.codes
inf['Irregular_menstrual_cycle'] = inf['Irregular_menstrual_cycle'].cat.codes
inf['Family_history'] = inf['Family_history'].cat.codes


# In[18]:


inf.head()


# In[19]:


X = inf.drop('Infertility',axis=1)
y = inf['Infertility']


# In[65]:


# Dataset is split to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=10)


# In[66]:


# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
test = SelectKBest(score_func=f_classif, k=8)
fit = test.fit(X_train, y_train.ravel())
features_name = test.get_support(indices=True)
# summarize scores
set_printoptions(precision=2)
print(fit.scores_)
print(features_name)
features = fit.transform(X_train)
# summarize selected features , number is the record number
print(features[0:6,:])


# In[80]:


l =[0,1,3,4,5,6,7,8]
col_names = list(X.columns)
for i in l:
    print(col_names[i])


# In[56]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[81]:


X_8f=inf[['First_period_age','Marriage_age','Family_history','Diabetes','Hypertension','Thyroid','Polycystic Ovary','Genital Infection']]





# In[82]:


X_8f_train, X_8f_test, y_train, y_test = train_test_split(X_8f, y, test_size=0.45, random_state=10)


# In[83]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
lr1 = LogisticRegressionCV(cv=5,random_state=10) 
lr1.fit(X_8f_train, y_train) 
predictions = lr1.predict(X_8f_test) 
print(confusion_matrix(y_test,predictions))  
# print classification report 
print(classification_report(y_test, predictions))


# In[84]:


# Trying Streamlit
import joblib
with open('model-v1.joblib', 'wb') as f:
    joblib.dump(lr1,f)


# In[85]:


def yes_or_no(value):
    if value == 'Yes':
        return 1
    else:
        return 0


# In[86]:


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
    st.markdown("""
<style>
body {
    color: #000;
    background-color: #CCC2BE;
}
</style>
    """, unsafe_allow_html=True)
    st.sidebar.subheader('Momma Details')
    F_PA_cat = st.sidebar.selectbox("Did you get your first period after the age of 15?",('No','Yes'))
    F_PA = yes_or_no(F_PA_cat)
    MA_cat = st.sidebar.selectbox("Did you marry after the age of 30?",('No','Yes'))
    MA = yes_or_no(MA_cat)
    FH_cat = st.sidebar.selectbox("Do you have a family history of Infertility?",('No','Yes'))
    FH = yes_or_no(FH_cat)
    Diabetes_cat = st.sidebar.selectbox("Are you diabetic?",('No','Yes'))
    Diabetes = yes_or_no(Diabetes_cat)
    HP_cat = st.sidebar.selectbox("Do you have Hypertension?",('No','Yes'))
    HP = yes_or_no(HP_cat)
    Th_cat = st.sidebar.selectbox("Have you had Thyroid issues?",('No','Yes'))
    Th = yes_or_no(Th_cat)
    PCOD_cat = st.sidebar.selectbox("Have you had PCOD problems?",('No','Yes'))
    PCOD = yes_or_no(PCOD_cat)
    GI_cat = st.sidebar.selectbox("Do you have Genital infection?",('No','Yes'))
    GI = yes_or_no(GI_cat)
    features = {'First_period_age':F_PA,
                'Marriage_age':MA,'Family_history':FH,
                'Diabetes':Diabetes,
                'Hypertension':HP,
                'Thyroid':Th,
                'Polycystic Ovary':PCOD,'Genital Infection':GI
               }
    data = pd.DataFrame(features,index=[0])
    st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#482473,#482473);
    color: white;
}
.Widget>label {
    color: white;
}
[class^="st-b"]  {
    color: black;
}
.st-at {
    background-color: white;
}


</style>
""",
    unsafe_allow_html=True,
)

    return data


# In[88]:


import streamlit as st
user_input_df = get_user_input()


# In[89]:

import streamlit as st
def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index=['Unlikely','Probably','Likely'])
    max_percentage = grad_percentage['Percentage'].max()
    result_index = grad_percentage.idxmax(axis = 0) 
    result = pd.DataFrame(data=max_percentage,columns = ['Percentage'],index = result_index)
    #my_colors = ['green','yellow','red']
    if result_index[0] == 'Unlikely':
        colour = 'green'
    elif result_index[0] == 'Probably':
        colour = 'yellow'
    elif result_index[0] == 'Likely':
        colour = 'red'
    ax = result.plot(kind='barh', figsize=(7, 4),zorder=10, width=0.1,color = colour,visible=True)
    ax.legend().set_visible(True) 
    ax.set_xlim(xmin=0, xmax=100) 
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False) 
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xticks([0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
       ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Percentage(%)", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Infertility Chance", labelpad=2, weight='bold', size=12)
    ax.set_title('Risk of Infertility ', fontdict=None, loc='center', pad=None, weight='bold')
    st.title(str(max_percentage) + " % : " + result_index[0] +" Risk ")
    st.pyplot()
    return


# In[90]:


st.set_option('deprecation.showPyplotGlobalUse', False)


# In[91]:


prediction_proba = lr1.predict_proba(user_input_df)
visualize_confidence_level(prediction_proba)

