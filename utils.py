from ultralytics import YOLO
from models.comparator.ResNet50_similarity_Xing import compare
from models.recipe.recipe_parser import findRecipes
from models.recipe.recipe_generate import generate_recipe
import cv2 as cv
import os, shutil

def removeFiles(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def runAI(frames):
    # Remove all Files in generatedFrames
    removeFiles('generatedFrames')
    # Parameters to Change
    confidenceLevelGeneral = 0.5
    confidenceLevelClassifier = 0.5
    confidenceLevelComparator = 0.7

    dataAcquired = []
    itemsToCompare = []
    for index, frame in enumerate(frames):
        # This is the General Ingredient AI
        model = YOLO("models/general/general.pt")
        results = model([frames[index]], conf=confidenceLevelGeneral)

        result = results[0]
        generatedImagePath = []

        for idx,box in enumerate(result.boxes.xyxy):
            x1,y1,x2,y2 = box.cpu().numpy().astype(int)
            fileName = f"generatedFrames/image{index}_{idx}.png"
            cv.imwrite(fileName, frames[index][y1:y2,x1:x2,:])
            generatedImagePath.append(fileName)

        modelClassifier = YOLO("models/classifier/best.pt")
        for path in generatedImagePath:
            resultClassifier = modelClassifier([path], conf=confidenceLevelClassifier)

            probs = resultClassifier[0].summary()
            if len(probs) > 0:
                for items in probs:
                    dataAcquired.append(items["name"])
            else:
                itemsToCompare.append(path)

    print(itemsToCompare)

    # Get all Items in compare Directory
    path = "compare"
    dir_list = os.listdir(path)

    comparisonItems = []
    for dir in dir_list: 
        comparisonItems.append([dir.replace(".jpg", "").replace("_", " "), dir])

    for imagesToCompare in itemsToCompare:
        for index, images in enumerate(comparisonItems):
            if compare("compare/" + images[1], imagesToCompare) > confidenceLevelComparator:
                dataAcquired.append(images[0])
    
    cleanDataAcquired = list(set(dataAcquired))
    foundRecipes = findRecipes(cleanDataAcquired)

    ingredientString = ""
    for items in cleanDataAcquired:
        ingredientString = ingredientString + str(dataAcquired.count(items)) + " " + str(items) + "\n"

    crudeGeneratedRecipes = generate_recipe(ingredientString)

    cooking_verbs = [
        "add", "bake", "barbecue", "baste", "beat", "blend", "boil", "braise", "bread",
        "break", "brew", "broil", "brown", "brush", "caramelize", "chill", "chop", "clarify",
        "coat", "combine", "cook", "cool", "cover", "cream", "crush", "cube", "cut", "debone",
        "deep-fry", "defrost", "deglaze", "dice", "dissolve", "drain", "dress", "drizzle",
        "dry", "fry", "fold", "freeze", "garnish", "glaze", "grate", "grill", "grind", "heat",
        "infuse", "julienne", "knead", "layer", "leaven", "marinate", "mash", "melt", "microwave",
        "mince", "mix", "pan-fry", "parboil", "peel", "poach", "pour", "preheat", "press",
        "puree", "refrigerate", "reduce", "roast", "roll", "saute", "scald", "scramble",
        "sear", "season", "serve", "shred", "sift", "simmer", "slice", "smoke", "soak",
        "soften", "sprinkle", "steam", "stir", "strain", "stuff", "tenderize",
        "toast", "toss", "trim", "wash", "whip", "whisk", "zest"
    ]
    cooking_prepositions = [
        "in", "on", "at", "into", "onto", "over", "under", "by",
        "from", "to", "for", "after", "before", "during", "around", "about",
        "between", "through", "across", "within", "without", "along", "underneath",
        "inside", "outside", "beneath", "beside", "above", "below", "near", "behind"
    ]

    markedIndex = []
    for index, words in enumerate(crudeGeneratedRecipes.split(" ")):
        for verbs in cooking_verbs:
            if words.lower() == verbs.lower():
                markedIndex.append(index)
        for prepositons in cooking_prepositions:
            if words.lower() == prepositons.lower():
                markedIndex.append(index)
    markedIndex.sort()

    title = ""
    instruction = ""
    for index, words in enumerate(crudeGeneratedRecipes.split(" ")):
        if index == markedIndex[0]:
            break
        title = title + words + " "
                
    for index, words in enumerate(crudeGeneratedRecipes.split(" ")):
        if index >= markedIndex[0]:
            instruction = instruction + words + " "        

    generatedRecipes = [title ,ingredientString, instruction.replace(".", ".\n").replace("\n ", "\n")]

    return ingredientString, foundRecipes, generatedRecipes