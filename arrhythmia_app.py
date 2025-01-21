import pandas as pd
import numpy as np
import streamlit as st
print("success1")
#st.title("AI-Enhanced ECG Arrhythmia Classification ")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from sklearn.utils import class_weight
import itertools
import tensorflow as tf
#from tensorflow.keras.models import Functional
import streamlit as st



custom_css = """
<style>
body {
    background-image: url("https://www.myvmc.com/wp-content/uploads/2013/11/heartbeat-stethoscope.jpg");
    background-size: cover;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.title("AI-Enhanced ECG Arrhythmia Classification ")
train_df=pd.read_csv("D:\mitbih_train.csv")
test_df=pd.read_csv("D:\mitbih_test.csv")
import pandas as pd
st.header('ECTOPIC BEATS')
st.write("An ectopic beat, also known as premature ventricular contraction (PVC) or premature atrial contraction (PAC), refers to an irregularity in the heart's rhythm. Non-ectopicly, the heart's electrical system generates rhythmic impulses that coordinate the heartbeats.")
html_temp="""
<div style="background-color:tomato;padding:10px">
<center><h2 style="color:white;">Arrhythmia Classifier </h2>
</center></div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
train_df.columns =[i for i in range(188)]
test_df.columns =[i for i in range(188)]
#st.write(train_df.head())
#st.write(test_df.head())

st.sidebar.header('User Inputs')
selected_analysis = st.sidebar.selectbox('Select your choice:', ['Test Your ECG','Study Different Ectopic Beats'])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:\\Users\\saray\\OneDrive\\Desktop\\mymodel_idt.h5")

mod = load_model()


if (selected_analysis=='Study Different Ectopic Beats'):
    
    selected_beats = st.sidebar.selectbox('choose type of arrhythmia:', ['Non-Ectopic Beats', 'Supraventricular Ectopic Beats','Venticular Ectopic Beats','Fusion Beats','Unknown Beats'])

    #to study the differente classes
    samp=train_df.groupby(187,group_keys=False).apply(lambda train_df : train_df.sample(1))
    #st.write(samp)
    
    def plot(i):
        x2= np.arange(186)
        fig2, axes2 = plt.subplots()
        
        axes2.plot(x2,samp.iloc[i,:186])
        
        axes2.set_title(selected_beats)
        axes2.set_xlabel("Time in decaseconds")
        axes2.set_ylabel("Voltage in milliVolts")
        st.pyplot(fig2)
    x = np.arange(186)
    
    
    
    if(selected_beats=='Non-Ectopic Beats'):
        st.header('NON-ECTOPIC BEATS')
        #st.write("A non-ectopic beat refers to a Non-Ectopic heart rhythm originating from the sinus node, the heart's natural pacemaker. In contrast to ectopic beats, which arise from abNon-Ectopic electrical impulses originating outside the sinus node, non-ectopic beats follow the typical electrical pathway in the heart.")
        #st.write("Non-Ectopic beats refer to the regular, coordinated contractions of the heart's atria and ventricles that occur in a healthy individual with a properly functioning cardiac conduction system.")
        st.write("Non-ectopic beats encompass a spectrum of cardiac rhythms arising from the non-ectopic conduction system but exhibiting non-ectopic electrical patterns. This category includes bundle branch blocks (BBBs), such as left bundle branch blocks (LBBB) and right bundle branch blocks (RBBB), which cause characteristic changes in the ECG waveform, notably widened QRS complexes. Additionally, non-ectopic beats may involve atrial and nodal escape rhythms, where alternative pacemaker sites in the atria or atrioventricular (AV) node initiate heartbeats in the absence of non-ectopic sinus node activity.")
        st.subheader("ECG recording of Non-Ectopic beats")
        plot(0)
        #st.subheader("Distribution of voltage values in the Non-Ectopic beats class")
        #n_e()
        st.write("<b>Non-ectopic Arrhythmia</b> comprises the following",unsafe_allow_html=True)
        st.write("<ul style='list-style-type: square;'>",unsafe_allow_html=True)
        st.write(" <li> <b>Left Bundle Branch Block (LBBB):</b>  A delay or blockage of electrical impulses along the left bundle branch of the heart's electrical conduction system, leading to an abnormal ECG pattern.  </li>",unsafe_allow_html=True)
        st.write(" <li> <b>Right Bundle Branch Block (RBBB):</b> Similar to LBBB but affecting the right bundle branch, resulting in characteristic changes in the ECG waveform.  </li>",unsafe_allow_html=True)
        st.write(" <li> <b>Atrial Escape:</b> A late beat originating from the atria when the sinus node fails to initiate an impulse.  </li>",unsafe_allow_html=True)
        st.write(" <li>  <b>Nodal Escape:</b> A late beat originating from the atrioventricular (AV) node when the sinus node fails to initiate an impulse. </li>",unsafe_allow_html=True)
        st.write("</ul>",unsafe_allow_html=True)
        
    elif(selected_beats== 'Supraventricular Ectopic Beats'):
        st.header('SUPRAVENTRICULAR ECTOPIC BEATS')
        #st.write('A supraventricular beat refers to an abNon-Ectopic heart rhythm originating from above the ventricles of the heart. This includes the atria (upper chambers of the heart) and the atrioventricular node (AV node), but not the ventricles (lower chambers of the heart).')
        st.write("Supraventricular ectopic arrhythmias encompass various abnormal heart rhythms originating above the ventricles. This category includes atrial premature beats, which arise from the atria prematurely activating before the next expected heartbeat. Additionally, aberrant atrial premature beats occur when abnormal electrical pathways within the atria lead to irregular contractions. Nodal premature beats originate from the atrioventricular node, while supraventricular premature beats originate from other supraventricular tissues.")
        st.subheader("ECG recording of supraventricular beats")
        plot(1)
        #st.subheader("Distribution of voltage values in the supraventricular beats class")
        #plot_hist(1, 50, 5, 45)
        #s_e()
        st.write("<b>Supraventricular Ectopic Arrhythmia</b> comprises the following",unsafe_allow_html=True)
        st.write("<ul style='list-style-type: square;'>",unsafe_allow_html=True)
        st.write(" <li> <b>Atrial Premature:</b> An early beat originating from the atria, occurring earlier than expected.  </li>",unsafe_allow_html=True)
        st.write(" <li> <b>Aberrant Atrial Premature:</b> An abnormal early beat originating from the atria with irregular conduction pathways.  </li>",unsafe_allow_html=True)
        st.write(" <li> <b> Nodal Premature:</b> An early beat originating from the AV node. </li>",unsafe_allow_html=True)
        st.write(" <li><b> Supraventricular Premature:</b> An early beat originating above the ventricles but below the atria.  </li>",unsafe_allow_html=True)
        st.write("</ul>",unsafe_allow_html=True)
        
    elif(selected_beats=='Venticular Ectopic Beats'):
        st.header('VENTRICULAR ECTOPIC BEATS')
        #st.write("Ventricular ectopic beats, also known as premature ventricular contractions (PVCs), occur when the heart's ventricles (the lower chambers) contract prematurely due to an abNon-Ectopic electrical impulse.")
        #st.write("Ventricular ectopic beats, also known as premature ventricular contractions (PVCs), are irregular heartbeats that originate in the ventricles of the heart. Non-Ectopicly, the heart's electrical system coordinates the timing of heartbeats, starting in the sinoatrial (SA) node and traveling through the atria and then the ventricles. However, in ventricular ectopic beats, an abNon-Ectopic electrical impulse occurs in the ventricles before the regular heartbeat initiated by the SA node.")
        st.write("Ventricular ectopic arrhythmias (Class V) encompass abnormal heart rhythms originating within the ventricles. This category includes premature ventricular contractions (PVCs), which occur when the heart's ventricles contract earlier than expected. Ventricular escape beats, another type within this class, occur when the heart's normal pacemaker fails and the ventricles take over the heart's rhythm temporarily. These arrhythmias can lead to sensations like palpitations, skipped beats, or a fluttering feeling in the chest.")
        st.subheader("ECG recording of ventricular beats")
        plot(2)
        #st.subheader("Distribution of voltage values in the ventricular beats class")
        #plot_hist(2, 50,10, 40)
        #v_e()
        st.write("<b>Ventricular Ectopic Arrhythmia</b> comprises the following",unsafe_allow_html=True)
        st.write("<ul style='list-style-type: square;'>",unsafe_allow_html=True)
        st.write(" <li><b> Premature Ventricular Contraction (PVC):</b> An early beat originating from the ventricles.  </li>",unsafe_allow_html=True)
        st.write(" <li> <b> Ventricular Escape:</b> A late beat originating from the ventricles when the normal pacemaker sites fail to initiate an impulse. </li>",unsafe_allow_html=True)
        st.write("</ul>",unsafe_allow_html=True)
    
        
    elif(selected_beats=='Fusion Beats'):
        st.header('FUSION BEATS')
        #st.write("Fusion beats are a type of cardiac arrhythmia that occur when a Non-Ectopic, sinus impulse coincides with an ectopic impulse in the heart's ventricles.")
        st.write("Fusion beats are a type of cardiac rhythm disturbance that occurs when a premature ventricular contraction (PVC) and a Non-Ectopic sinus beat coincide in such a way that they produce a hybrid QRS complex on an electrocardiogram (ECG). In other words, a fusion beat occurs when the premature electrical impulse from the ventricle coincides with the Non-Ectopic electrical impulse from the sinus node, resulting in a complex waveform that has characteristics of both Non-Ectopic and premature beats.")
        st.subheader("ECG recording of fusion beats")
        plot(3)
        #st.subheader("Distribution of voltage values in the fusion beats class")
        #plot_hist(3, 60, 15, 45)
        #f_e()
        st.write("<b>Fusion Beats</b> comprises the following", unsafe_allow_html=True)
        st.write("<ul style='list-style-type: square;'>",unsafe_allow_html=True)
        st.write(" <li> <b> Fusion of Ventricular and Non-ectopic Beats:</b> These beats occur when an electrical impulse originating from the ventricles coincides with a normal non-ectopic beat originating from the sinus node or atria. </li>",unsafe_allow_html=True)
        st.write(" <li> <b>Fusion of Ventricular and Supraventricular Beats:</b> Fusion beats can also occur when an electrical impulse from the ventricles coincides with an early beat originating from above the ventricles (supraventricular premature beat). This results in a hybrid waveform that combines characteristics of both types of beats.  </li>",unsafe_allow_html=True)
        st.write("</ul>",unsafe_allow_html=True)
        

    else:
        st.header('UNKNOWN BEATS')
        #st.write("Unknown beats typically refer to instances in electrocardiogram (ECG) recordings where the origin or nature of a particular heartbeat is unclear or cannot be definitively identified based on standard criteria. In other words, they are beats that do not fit into recognized categories such as Non-Ectopic sinus rhythm, ectopic beats, or fusion beats.")
        st.write("The unknown beats of unknown origin, including paced beats and the fusion of paced and normal beats. Paced beats occur when the heart's rhythm is artificially controlled by an external pacemaker, which emits electrical impulses to regulate the heartbeat. Fusion beats in this category arise from the interaction between the pacemaker-generated impulses and the heart's intrinsic electrical activity.")
        st.subheader("ECG recording of unknown beats")
        plot(4)
        #st.subheader("Distribution of voltage values in the unknown beats class")
        #plot_hist(4, 50, 15, 35)
        #u_e()
        st.write("<b>Unknown Beats</b> comprises the following",unsafe_allow_html=True)
        st.write("<ul style='list-style-type: square;'>",unsafe_allow_html=True)
        st.write(" <li> <b>Paced Beats:</b> Beats artificially stimulated by an external pacemaker device.  </li>",unsafe_allow_html=True)
        st.write(" <li> <b>Fusion of Paced and Normal Beats:</b> Beats that are a combination of artificially paced and normal beats.s  </li>",unsafe_allow_html=True)
        st.write("</ul>",unsafe_allow_html=True)
        

    
else:#if(selected_analysis=='test your ECG'):
    st.header("Test Your ECG recording")

    # Add a file uploader widget
    file = st.file_uploader("Upload CSV File", type=["csv"])

    # Check if a file has been uploaded
    if file is not None:
        # Read the Excel file into a DataFrame
        #if file is type=="xlsx" or type=="xls" :
        #    df = pd.read_excel(file,header=None, encoding='latin1')
        #else:
        df=pd.read_csv(file,header=None)
        df.columns =range( len(df.columns) )

        # Display the DataFrame
        st.write(df)
        x3= np.arange(len(df.columns))
        fig3, axes3 = plt.subplots()
        axes3.plot(x3,df.iloc[0,:])
        axes3.set_title("Your ECG Recording")
        axes3.set_xlabel("Time in decaseconds")
        axes3.set_ylabel("Voltage in milliVolts")
        st.pyplot(fig3)
    else:
        st.write("Upload an appropriate file to test your ECG recording.")
        st.write("Make sure the recordings are digitized at 360 samples per second per channel with an 11-bit resolution over a 10-millivolt range.")

        




if(selected_analysis=='Test Your ECG'):
    if (file):
        #st.line_chart(df)
        df=df.iloc[:,:186].values
        df = df.reshape(len(df), df.shape[1],1)
        y_pred1=mod.predict(df)
        y_pred1=y_pred1.argmax(axis=1)
        #st.subheader("The predicted classification of the given ECG recording:")
        #st.write("Classification Number",y_pred1)
        if (y_pred1==0):
            y_pred2='Non-Ectopic Beats'
            st.metric(label="The predicted classification of the given ECG recording:",value=f"{y_pred2}")
            #st.markdown("<p style='color: blue;'> Your ECG recording is classified to be <b>Non-Ectopic beats</b>.</p>",unsafe_allow_html=True)
            #st.write("Your ECG recording is classified to be Non-Ectopic beats")
            st.write("Non-Ectopic beats are typically categorized to be not life threatening. In order to avoid more severe issues, it might need to be addressed.")
            #st.write("The ECG is predicted with 92.35'%' accuracy. There is 8 percent chance of being wrong. ")
        elif (y_pred1==1):
            y_pred2= 'Supraventricular Ectopic Beats'
            st.metric(label="The predicted classification of the given ECG recording:",value=f"{y_pred2}")
            #st.markdown("<p style='color: blue;'> Your ECG recording is classified to be <b>Supraventricular Ectopic beats</b>.</p>",unsafe_allow_html=True)
            #st.write("Your ECG recording is classified to be Supraventricular beats")
            st.write("Supraventricular ectopic beats are typically categorized as high risk since they raise the possibility of stroke or cardiac arrest.")
        elif (y_pred1==2):
            y_pred2='Venticular Ectopic Beats'
            st.metric(label="The predicted classification of the given ECG recording:",value=f"{y_pred2}")
            #st.markdown("<p style='color: blue;'> Your ECG recording is classified to be <b>Ventricular Ectopic beats</b>.</p>",unsafe_allow_html=True)
            #st.write("Your ECG recording is classified to be Ventricular beats")
            st.write("Ventricular ectopic beats are typically categorized as medium risk since they raise the possibility of an arrhythmia or lower cardiac output.")
            #st.write("Ventricular ectopic beats are generally classified to be medium risk, as it may increases the risk of arrhythmia or reduce cardiac output.")
        elif (y_pred1==3):
            y_pred2='Fusion Beats'
            st.metric(label="The predicted classification of the given ECG recording:",value=f"{y_pred2}")
            #st.markdown("<p style='color: blue;'> Your ECG recording is classified to be <b>Fusion beats</b>.</p>",unsafe_allow_html=True)
            #st.write("Your ECG recording is classified to be Fusion beats")
            st.write("Fusion beats are typically categorized as medium risk since they have the potential to impact heart function or raise the risk of arrhythmia.")
        else :
            y_pred2='Unknown Beats'
            #st.line_chart(y_pred1)
            st.metric(label="The predicted classification of the given ECG recording:",value=f"{y_pred2}")
            #st.markdown("<p style='color: blue;'> Your ECG recording is classified to be <b>Unknown beats</b>.</p>",unsafe_allow_html=True)
            #st.write("Your ECG recording is classified to be Unknown beats. ")
            st.write("Unknown beats are typically categorized as high risk since they raise the possibility of a serious arrhythmia or leads to cardiovascular complications.")
print("success1")