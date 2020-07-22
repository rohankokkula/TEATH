# TEATH-Text Extraction And THresholding
## TEATH is a WebApp for Extracting Text from Images and Thresholding with the help of pytesseract and OpenCV.
### This project is an app-version of ML challenge by HackerEarth Pride Month Edition<br>Ranked 110th Position out of 5000 Participants.
Link to Competition : https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-pride-month-edition/<br>
## Try it out! teath.herokuapp.com
Steps to Follow:
1. Clone the repository `git clone https://github.com/rohankokkula/teath.git`
2. For Localhost:
    1. `pip install -r requirements.txt`
    2. Install tesseract executable from https://github.com/UB-Mannheim/tesseract/wiki, <br>keep an eye on installation path.
    3. Open app.py, change tesseract_cmd path to installation path's executable file(eg. C:\Users\rohan\AppData\Local\Tesseract-OCR\tesseract.exe)
    4. Run cmd in current folder and enter `streamlit run app.py`
    5. App will be deployed at localhost:8501(mostly)
3. For Heroku Deployment:
    1. Create Heroku account and Install Heroku CLI from here -> https://devcenter.heroku.com/articles/heroku-cli#download-and-install
    2. Create APP from Heroku Dashboard.
    3. Go to your APP dashboard settings on heroku website and in the buildpacks URL,<br>enter `https://github.com/heroku/heroku-buildpack-apt`<br>
    Now, Reveal Config vars and add <br>KEY: `TESSDATA_PREFIX`<br>VALUE:`./.apt/usr/share/tesseract-ocr/4.00/tessdata`
    3. Run cmd in current folder and enter `heroku login`( Logging into your account)
    4. After successful login, follow the below steps:
    ```
    git add .
    git commit -am "First commit"
    heroku git:remote -a app-name
    git push heroku master
    ```
    5. App will be deployed at app-name.herokuapp.com

<br>
Link to youtube Tutorial: <br>

References:
1. Streamlit: https://docs.streamlit.io/en/stable/api.html
2. OpenCV Thresholding: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
3. Pytesseract: https://pypi.org/project/pytesseract/ 
4. Tesseract deployment on Heroku using Flask: https://towardsdatascience.com/deploy-python-tesseract-ocr-on-heroku-bbcc39391a8d
