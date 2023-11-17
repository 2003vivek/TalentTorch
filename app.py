import pandas as pd
import re
import streamlit as st
import nltk 
import pickle
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
nltk.download('punkt')
words=nltk.download('stopwords')

clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

df=pd.read_csv('UpdatedResumeDataSet.csv')
df['category_names']=df['Category']
df['Category']=le.fit_transform(df['Category'])
label_mapping = dict(zip(df['Category'],df['category_names']))

def cleanData(txt):
    cleandata=re.sub('http\S+', '', txt)

    # data=cleandata.split()
    # for i in data:
    #     if i in words:
    #         cleandata=cleandata.replace(i,"")
    cleandata=re.sub('[^a-zA-Z0-9\s]','',cleandata)
    cleandata=re.sub('\s+',' ',cleandata)
    cleandata=re.sub('\d',' ',cleandata)
    
    
    return cleandata
def main():
    custom_css = """
    <style>
        /* Add your Bootstrap styles or custom CSS here */
        body {
            font-family: 'Arial', sans-serif;
            background-color:black;
            color:black;
        }
        .stApp {
            background-color: gray;
            color:black;
        }
        
        .navbar {
            background-color: #333;
            padding: 10px;
            color: white;
            text-align: center;
        }

        /* Style the navigation links */
        .navbar a {
            color: white;
            padding: 8px 16px;
            text-decoration: none;
            display: inline-block;
        }
        
    </style>
    """

    # Display the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown('<div class="navbar">\
                <a href="#home">Home</a>\
                <a href="#about">About</a>\
                <a href="#contact">Contact</a>\
            </div>', unsafe_allow_html=True)
    st.title('Resume-Screening')
    
    # st.image("https://d.novoresume.com/images/doc/minimalist-resume-template.png",'Resumes',400)
    # st.write(" ")
    # st.image("https://d.novoresume.com/images/doc/minimalist-resume-template.png",'Resumes',400)
    image1 = "https://d.novoresume.com/images/doc/minimalist-resume-template.png"
    image2 = "https://www.jobhero.com/resources/wp-content/uploads/2023/07/tutor-template-resume-JH.svg"
    image3 = "https://www.greatsampleresume.com/wp-content/themes/resumebaking/img/home-carou/original.png"

    # Create columns for layout
    col1, col2, col3 = st.columns(3)

    # Display images in each column
    with col1:
        st.image(image1 )
        st.write(" ")

    with col2:
        st.image(image2)
        st.write(" ")

    with col3:
        st.image(image3 )
        st.write(" ")
    upload=st.file_uploader('UploadResume',type=['pdf','txt','word','doc'])
    

    if upload is not None:
        try:
            resume_bytes=upload.read()
            resume_txt=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_txt=resume_bytes.decode('latin-1')
    
        clean_res=cleanData(resume_txt)
        clean_resume=tfidf.transform([clean_res])
        prediction_id=clf.predict(clean_resume)[0]
        prediction=label_mapping.get(prediction_id,'unknown')
        st.write(" This Resume is suited for :",prediction)



if __name__=="__main__":
    main()
