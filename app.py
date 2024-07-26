from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from geopy.distance import geodesic
import googlemaps
from sklearn.cluster import DBSCAN

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Initialize the Google Maps client
API_KEY = 'AIzaSyD7ifW_3VBgfX1baabxSM5dlPtInK9a2bQ'
gmaps = googlemaps.Client(key=API_KEY)
 
def fetch_amenity_coordinates(location, amenity_name, radius): # give coordinates
    try:
        places_result = gmaps.places_nearby(location=location, radius=radius, keyword=amenity_name)
        amenities_info = [
            {
                'name': place['name'],
                'address': place.get('vicinity', 'No address available'),
                'rating': place.get('rating', 0),
                'location': (place['geometry']['location']['lat'], place['geometry']['location']['lng'])
            }
            for place in places_result.get('results', [])
        ]
        return amenities_info
    except googlemaps.exceptions.ApiError as e:
        print(f"API Error: {e}")
        return []

def calculate_distance(point1, point2): #calculate the distance between two points
    return geodesic(point1, point2).kilometers

def get_travel_time(origin, destination): # It find out the time taking between origin to destination
    try:
        directions_result = gmaps.directions(origin, destination, mode="driving")
        if directions_result:
            duration = directions_result[0]['legs'][0]['duration']['value']  # duration in seconds 
            return duration / 60  # convert to minutes
        else:
            return float('inf')
    except googlemaps.exceptions.ApiError as e:
        print(f"API Error: {e}")
        return float('inf')

def score_location(location, amenities, weights, current_location, time_preference, rating_preference, max_distance, max_time):
    score = 0
    time_weight = time_preference / (time_preference + rating_preference) if (time_preference + rating_preference) != 0 else 0
    rating_weight = rating_preference / (time_preference + rating_preference) if (time_preference + rating_preference) != 0 else 0
    
    travel_time = get_travel_time(current_location, location) / max_time  # Normalize travel time
    if travel_time == float('inf'):
        return float('inf')

    for amenity, details in amenities.items():
        distances = np.array([calculate_distance(location, detail['location']) for detail in details]) / max_distance  # Normalize distances
        ratings = np.array([detail['rating'] for detail in details])
        
        if distances.size > 0:
            min_distance = distances.min()
            max_rating = ratings.max()
            score += weights[amenity] * (time_weight * (min_distance + travel_time) - rating_weight * max_rating)
    
    return score

def get_address_from_coordinates(coordinates):
    try:
        result = gmaps.reverse_geocode(coordinates)
        if result:
            return result[0]['formatted_address']
        else:
            return "No address found"
    except googlemaps.exceptions.ApiError as e:
        print(f"API Error: {e}")
        return "Error retrieving address"

def find_clusters(amenities):
    locations = []
    for amenity, details in amenities.items():
        for detail in details:
            locations.append(detail['location'])
    
    if not locations:
        return []
    
    dbscan = DBSCAN(eps=0.01, min_samples=5)
    dbscan.fit(locations)
    unique_labels = set(dbscan.labels_)
    cluster_centers = []
    for label in unique_labels:
        if label != -1:
            cluster_points = [locations[i] for i in range(len(locations)) if dbscan.labels_[i] == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
    
    return cluster_centers

@app.route('/find_best_location', methods=['POST'])
def find_best_location():
    data = request.get_json()
    current_location = tuple(map(float, data['current_location'].split(',')))
    city = data['city']
    country = data['country']
    best_location_type = data['best_location_type']
    time_preference = float(data['time_preference'])
    rating_preference = float(data['rating_preference'])
    min_rating = float(data['min_rating'])
    amenities_list = data['amenities_list']
    radius = float(data.get('radius', 5000))  # Default radius is 5000 meters if not provided
    
    amenities = {}
    weights = {}
    equal_weight = 1.0 / len(amenities_list)

    full_address = f"{city}, {country}"
    city_location = gmaps.geocode(full_address)[0]['geometry']['location']
    city_coordinates = (city_location['lat'], city_location['lng'])

    for amenity_name in amenities_list:
        locations = fetch_amenity_coordinates(city_coordinates, amenity_name, radius)
        if locations:
            amenities[amenity_name] = locations
            weights[amenity_name] = equal_weight
        else:
            print(f"No {amenity_name} locations found.")

    if not amenities:
        return jsonify({"error": "No amenities found."}), 400

    best_location_candidates = fetch_amenity_coordinates(city_coordinates, best_location_type, radius)
    if not best_location_candidates:
        return jsonify({"error": f"No locations found for the specified type: {best_location_type}."}), 400

    if min_rating > 0:
        best_location_candidates = [loc for loc in best_location_candidates if loc['rating'] != 0 and loc['rating'] >= min_rating]
        if not best_location_candidates:
            return jsonify({"error": f"No locations found for the specified type: {best_location_type} with a rating higher than {min_rating}."}), 400

    cluster_centers = find_clusters(amenities)
    if not cluster_centers:
        return jsonify({"error": "No clusters found."}), 400

    max_distance = max([calculate_distance(current_location, tuple(center)) for center in cluster_centers])
    max_time = max([get_travel_time(current_location, tuple(center)) for center in cluster_centers])

    best_score = float('inf')
    best_location = None

    for center in cluster_centers:
        score = score_location(tuple(center), amenities, weights, current_location, time_preference, rating_preference, max_distance, max_time)
        if score < best_score:
            best_score = score
            best_location = tuple(center)

    if best_location is None:
        return jsonify({"error": "No suitable location found."}), 400

    best_location_address = get_address_from_coordinates(best_location)
    
    # Get the travel time from the chosen best location to the current location
    try:
        travel_time_response = gmaps.directions(best_location, current_location, mode="driving")
        travel_time_to_current_location = travel_time_response[0]['legs'][0]['duration']['text']
    except googlemaps.exceptions.ApiError as e:
        print(f"API Error: {e}")
        travel_time_to_current_location = "Error retrieving travel time"

    response = {
        "best_location_address": best_location_address,
        "best_location_score": best_score,
        "travel_time_to_current_location": travel_time_to_current_location,
        "distances_to_amenities": []
    }

    for amenity, details in amenities.items():
        closest_detail = min(details, key=lambda detail: calculate_distance(best_location, detail['location']))
        distance = calculate_distance(best_location, closest_detail['location'])
        response["distances_to_amenities"].append({
            "amenity": amenity,
            "distance_km": distance,
            "name": closest_detail['name'],
            "address": closest_detail['address'],
            "rating": closest_detail['rating']
        })

    best_location_detail = min(best_location_candidates, key=lambda detail: calculate_distance(best_location, detail['location']))
    
    # Include travel time to the chosen best location in the response
    try:
        travel_time_response = gmaps.directions(current_location, best_location_detail['location'], mode="driving")
        travel_time_to_best_location = travel_time_response[0]['legs'][0]['duration']['text']
    except googlemaps.exceptions.ApiError as e:
        print(f"API Error: {e}")
        travel_time_to_best_location = "Error retrieving travel time"
    
    response["chosen_best_location"] = {
        "name": best_location_detail['name'],
        "address": best_location_detail['address'],
        "rating": best_location_detail['rating'],
        "time_to_travel": travel_time_to_best_location
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
