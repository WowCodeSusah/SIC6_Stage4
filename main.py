from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
collectedData = []

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/receiveDetected")
async def receiveAPI():
    return {"ingredient": collectedData[-1][0]}

@app.get("/receiveFoundRecipes")
async def receiveRecipe():
    availableRecipe = []
    for number, recipe in enumerate(collectedData[-1][1]):
        availableRecipe.append(recipe)
        if number == 5:
            break

    return {"recipes": availableRecipe}

@app.get("/receiveGeneratedRecipes")
async def receiveGenerated():
    return {"generatedRecipes": collectedData[-1][2]}

from utils import runAI
from getFrame import getFrame
import cv2 as cv

def getResults():
    ESP32_URL = ["http://192.168.1.26:81/stream", "http://192.168.1.27:81/stream"]
    frames = getFrame(ESP32_URL)

    dataAcquired, foundRecipes, generatedRecipes = runAI(frames)
    print(dataAcquired)
    print(foundRecipes)
    print(generatedRecipes)

    return [dataAcquired, foundRecipes, generatedRecipes]

import threading
def run_function():
    thread = threading.Timer(300.0, run_function) # 300 seconds = 5 minute
    thread.start()
    collectedData.append(getResults())

run_function()