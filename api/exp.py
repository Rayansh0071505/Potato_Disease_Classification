# from fastapi import FastAPI, File,UploadFile
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
#
# # TO get the define function of fastapi type localhost:8000/docs
#
# app = FastAPI()
#
# @app.get("/ping")
# async def ping():
#     return "Hello! It's working"
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
# @app.post("/predict")
#
# # here we are uploading image as file thats why we type here
#
# async def predict_model(
#     file: UploadFile = File(...)
#         # upload file is file type type fastapi upload files in browser for more info
#
# ):
#     image = read_file_as_image(await file.read())
#     # watch use of await and async and see how is it important
#     return image
#
#
# # here we have two method to upload files--
# # 1) using FastAPI here we go to browser type localhost:8000/docs
# # in that we get our predict subtab where it will come option file where we upload it as File
#
# # 2) another step using postman api where
# # first we open app -> then change get option to post option -> type https://localhost:8000/predict (predict we use because here we have chose app.post('/predict as name'))
# # -> then go to key type file (as here we have chose file as name for ex if we write xyz we have to write key as xyz )-> then at value select it as file type
# # -> upload the file (here we are uploading image type)-> then click send option -> then comeback to code and run you will find value has been send to server
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)


from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from flask import request, render_template



app = FastAPI()
# endpoint = "https://localhost:8501/v1/models/potatoes_disease_model:predict"

origins = [
    "http://localhost",
    # "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The code snippet you provided is using FastAPI's `add_middleware` function to add a middleware component called `CORSMiddleware` to your FastAPI application. Let's break down what each parameter is doing:

# - `CORSMiddleware`: This is the middleware class provided by FastAPI to handle Cross-Origin Resource Sharing (CORS) issues. CORS is a security feature implemented by web browsers to control whether web applications running at one origin (domain) can request resources from another origin. This middleware helps control which origins are allowed to access your API resources.

# - `allow_origins`: This parameter specifies the list of origins (domains) that are allowed to access your API. In the code snippet, the list includes `"http://localhost"` and `"http://localhost:3000"`, which are the origins allowed to make requests to your API. These should match the domains of the applications that will be consuming your API.

# - `allow_credentials`: This parameter is a boolean that indicates whether the API can include credentials (like cookies, HTTP authentication, etc.) in the requests. Setting it to `True` means that your API can handle requests that include credentials.

# - `allow_methods`: This parameter specifies which HTTP methods are allowed for requests from the specified origins. In this case, `["*"]` means that all HTTP methods (GET, POST, PUT, DELETE, etc.) are allowed.

# - `allow_headers`: This parameter specifies which HTTP headers are allowed in the requests from the specified origins. `["*"]` indicates that all headers are allowed.

# By setting up these parameters, you are configuring the CORS middleware to allow requests from the specified origins, with credentials enabled, allowing all HTTP methods, and allowing all headers. This is useful when you have a web application (such as a frontend) hosted on a different domain that needs to make API requests to your FastAPI backend.

# Keep in mind that while allowing all origins (`"*"`) might be suitable for development and testing, in a production environment, you should be more restrictive and only allow specific origins that you trust. This helps prevent potential security vulnerabilities.

pred_MODEL = tf.keras.models.load_model("C:/Users/Amit/Desktop/others/Data science/Projects/potato disease prediction/Models/2")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    if request.method == "GET":
	    return render_template("index.html")
    # return "Hello, I am alive"



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = pred_MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

# second commit