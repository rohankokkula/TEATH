<h1>TEATH - Text Extraction And THresholding</h1>
<a href="http://www.youtube.com/watch?v=C6BM_dg75ao" target="_blank">Watch demo</a>
[![Demonstration on Youtube](http://img.youtube.com/vi/C6BM_dg75ao/0.jpg)](http://www.youtube.com/watch?v=C6BM_dg75ao "Youtube video demonstration")
## A Streamlit-Heroku WebApp for Extracting Text from Images and applying various thresholding methods using Pytesseract and OpenCV.
### Ranked 113th Position out of 5042 Participants in HackerEarth's Pride Month Challenge.

<a href="https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-pride-month-edition/" target="_blank">Link to Competition</a><br>

## Try the app. It's live here.. http://teath.herokuapp.com

Wanna try on your own and make changes?<br>
Feel free to fork/clone!<br>
Follow these steps.
1. Clone the repository `git clone https://github.com/rohankokkula/teath.git`
2. For Localhost:
    1. `pip install -r requirements.txt`
    2. Install tesseract executable from https://github.com/UB-Mannheim/tesseract/wiki, <br>keep an eye on installation path.
    3. Open app.py, change tesseract_cmd path to installation path's executable file(eg. C:\Users\rohan\AppData\Local\Tesseract-OCR\tesseract.exe)
    4. Run cmd in current folder and enter `streamlit run app.py`
    5. App will be deployed at localhost:8501(mostly)
3. For Heroku Deployment:
    1. Create Heroku account and Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli#download-and-install
    2. Create APP from Heroku Dashboard.
    3. Go to your APP dashboard settings on heroku website and in the buildpacks URL,<br>enter `https://github.com/heroku/heroku-buildpack-apt`<br>
    Now, Reveal Config vars and add <br>KEY: `TESSDATA_PREFIX`<br>VALUE:`./.apt/usr/share/tesseract-ocr/4.00/tessdata`
    3. Run cmd in current folder and enter `heroku login`( Logging into your account)
    4. After successful login, follow these steps:
    ```
    git add .
    git commit -am "First commit"
    heroku git:remote -a app-name
    git push heroku master
    ```
    5. App will be deployed at app-name.herokuapp.com

<br>

References:
1. Streamlit: https://docs.streamlit.io/en/stable/api.html
2. OpenCV Thresholding: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
3. Pytesseract: https://pypi.org/project/pytesseract/ 
4. TextBlob: https://pypi.org/project/textblob/
5. Tesseract deployment on Heroku using Flask: https://towardsdatascience.com/deploy-python-tesseract-ocr-on-heroku-bbcc39391a8d
