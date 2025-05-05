import pandas as pd

df = pd.read_csv("models/recipe/combined_recipes.csv")

def findRecipes(desired_ingredients):
    # Normalize the ingredients column (lowercase everything for easier matching)
    df['Ingredients'] = df['Ingredients'].str.lower()

    # Filter step-by-step through the list of ingredients
    filtered_df = df.copy()

    for ingredient in desired_ingredients:
        filtered_df = filtered_df[filtered_df['Ingredients'].str.contains(ingredient, na=False)]

    recipes = []
    for _, row in filtered_df.iterrows():
        recipes.append([row['Title'], row['Ingredients'], row['Instructions']])

    return recipes