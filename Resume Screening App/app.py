import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

def cleanResume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# website
def main():
    st.title("Resume Screening App")
    uploadedFile = st.file_uploader('Upload Resume : ', type=['txt','pdf'])
    if uploadedFile is not None:
        try:
            resumeBytes = uploadedFile.read()
            resumeText = resumeBytes.decode('utf-8')
        except UnicodeDecodeError:
            resumeText = resumeBytes.decode('latin-1')
            
        cleanedResume = cleanResume(resumeText)
        inputFeatures = tfidfd.transform([cleanedResume])
        predictionId = clf.predict(inputFeatures)[0]
        st.write(predictionId)
        
        # Map category ID to category name
        categoryMapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        categoryName = categoryMapping.get(predictionId, "Unknown")
        st.write("Predicted Category : ", categoryName)
  
if __name__=="__main__":
    main()

# To run the app run this command in the terminal : streamlit run app.py