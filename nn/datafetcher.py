import requests
import time
import csv
import os
import pandas as pd


API_KEY = "AIzaSyCo4C8j7kGJNFnr4hjK3KrANonXc5Dq56c"

def generateUrlToRating(filepath):
    df = pd.read_csv(filepath)
    urlPlusRating = []

    # in each row, we will store the first 3 images with rating
    for index, row in df.iterrows():
        place_id = row['id']
        rating = row['Rating']
        num_rating = row['User Ratings']

        details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=photos&key={API_KEY}"
        details_response = requests.get(details_url)
        details_data = details_response.json()

        if "photos" in details_data.get("result", {}):
            photos = details_data["result"]["photos"]
            
            # Extract first 3 photos (if available)
            for i, photo in enumerate(photos[:3]):
                photo_reference = photo["photo_reference"]
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photo_reference={photo_reference}&key={API_KEY}"
                urlPlusRating.append({"url": photo_url, "num_rating": num_rating, "rating": rating})
    resultdf = pd.DataFrame(urlPlusRating)
    resultdf.to_csv("urlToRating.csv", index=False)
    print("saved url + rating file!")





# cleans the retrieved locations by removing duplicates and no ratings
def cleanData(filepath):
    with open(filepath, "r", newline="", encoding="utf-8") as file:

        unique_entries = set()
        cleaned_data = []
        reader = csv.reader(file)
        header = next(reader) 
        cleaned_data.append(header)

        for row in reader:
            row_tuple = tuple(row)  
            if row_tuple not in unique_entries and row[4] != "No rating" and row[4] != "0":
                unique_entries.add(row_tuple)
                cleaned_data.append(row)

    cleaned_filepath = os.path.splitext(filepath)[0] + "_cleaned.csv"
    with open(cleaned_filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(cleaned_data)


# finds the viewpoints given a square area
def viewpointCoordinateFinder():
    radius = 50000  # 50km max radius

    # California Boundaries
    lat_min, lat_max = 32.5, 42.0  
    lng_min, lng_max = -124.4, -114.1 

    lat_step = 0.5
    lng_step = 0.5

    output_file = "california_viewpoints.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "id", "Latitude", "Longitude", "Rating", "User Ratings", "Address"])

        lat = lat_min
        while lat <= lat_max:
            lng = lng_min
            while lng <= lng_max:
                print(f"Searching at {lat}, {lng}")  

                url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&keyword=viewpoint&key={API_KEY}"
                response = requests.get(url)
                data = response.json()

                if data["status"] == "OK":
                    for place in data["results"]:
                        name = place.get("name", "Unknown")
                        rating = place.get("rating", "No rating")
                        user_ratings = place.get("user_ratings_total", "No ratings count")
                        place_lat = place["geometry"]["location"]["lat"]
                        place_lng = place["geometry"]["location"]["lng"]
                        id = place["place_id"]
                        address = place.get("vicinity", "Unknown location")

                        writer.writerow([name, id, place_lat, place_lng, rating, user_ratings, address])

                time.sleep(0.25)

                lng += lng_step

            lat += lat_step

    print(f"Data collection complete! Saved to {output_file}")



