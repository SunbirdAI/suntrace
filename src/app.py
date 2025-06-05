import os
import json
import sys
from flask import Flask, render_template, request, jsonify
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from dotenv import load_dotenv
from utils.llm_function_caller import ask_with_functions
from utils.factory import create_geospatial_analyzer

# Load environment variables from .env file
load_dotenv()



app = Flask(__name__)

# Initialize analyzer once
geospatial_analyzer = create_geospatial_analyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_map_layers')
def get_map_layers():
    """Return GeoJSON data for the map layers"""
    try:
        # Get bounds for the map from the plain tiles
        bounds = geospatial_analyzer._plain_tiles_gdf.total_bounds.tolist()
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]  # [lat, lon]

        # Convert minigrids to GeoJSON
        minigrids_geo = json.loads(geospatial_analyzer._minigrids_gdf.to_crs('EPSG:4326').to_json())
        
        # Get a sample of buildings to avoid performance issues (limit to 2000)
        building_sample = geospatial_analyzer._buildings_gdf.sample(min(2000, len(geospatial_analyzer._buildings_gdf))) if len(geospatial_analyzer._buildings_gdf) > 2000 else geospatial_analyzer._buildings_gdf
        buildings_geo = json.loads(building_sample.to_crs('EPSG:4326').to_json())
        
        return jsonify({
            'center': center,
            'bounds': [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],  # [[minLat, minLon], [maxLat, maxLon]]
            'minigrids': minigrids_geo,
            'buildings': buildings_geo
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query():
    """Handle LLM queries and process drawn polygons"""
    try:
        data = request.json
        user_query = data.get('query', '')
        
        # Process the drawn polygon if provided
        if 'polygon' in data:
            # Convert the polygon coordinates to a WKT string
            coords = data['polygon']
            # Create a shapely polygon from the coordinates
            polygon = Polygon(coords)
            # Convert the polygon to WKT format
            wkt_polygon = wkt_dumps(polygon)
            
            # Append the WKT polygon to the user query
            user_query = f"{user_query} Here's the region WKT: {wkt_polygon}"
        
        # Call the LLM function to process the query
        response = ask_with_functions(user_query, geospatial_analyzer)
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)