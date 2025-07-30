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
from utils.VisualizationGenerator import VisualizationGenerator
import openai
from typing import Optional, Dict

# Clear any existing problematic environment variable
if 'OPENAI_API_KEY' in os.environ: 
    del os.environ['OPENAI_API_KEY']
# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize analyzer once
geospatial_analyzer = create_geospatial_analyzer()
# Initialize the visualization generator
#visualization_generator = VisualizationGenerator(geospatial_analyzer)

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
        candidate_minigrids_geo = json.loads(geospatial_analyzer._candidate_minigrids_gdf.to_crs('EPSG:4326').to_json())
        
        # Get a sample of buildings to avoid performance issues (limit to 2000)
        building_sample = geospatial_analyzer._buildings_gdf.sample(min(2000, len(geospatial_analyzer._buildings_gdf))) if len(geospatial_analyzer._buildings_gdf) > 2000 else geospatial_analyzer._buildings_gdf
        buildings_geo = json.loads(building_sample.to_crs('EPSG:4326').to_json())
        
        return jsonify({
            'center': center,
            'bounds': [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],  # [[minLat, minLon], [maxLat, maxLon]]
            'candidate_minigrids': candidate_minigrids_geo,
            'buildings': buildings_geo
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Add a new route for handling visualization requests
# @app.route('/api/visualize', methods=['POST'])
# def visualize():
#     """Generate a visualization layer based on the query and region"""
#     try:
#         data = request.json
#         query_type = data.get('query_type')
#         parameters = data.get('parameters', {})
#         
#         if 'polygon' in data:
#             # Convert the polygon coordinates to a shapely polygon
#             coords = data['polygon']
#             polygon = Polygon(coords)
#             parameters['region'] = polygon.__geo_interface__
#         
#         # Generate the visualization layer
#         result = visualization_generator.generate_layer(query_type, parameters)
#         
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

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
        
        # Process the query to see if visualization is requested
#        visualization_request = detect_visualization_request(user_query)
      
        # Call the LLM function to process the query
        response = ask_with_functions(user_query, geospatial_analyzer)

        result = {'response': response}

        # If visualization is requested, add visualization metadata
#        if visualization_request:
#            result['visualization'] = {
#                'query_type': visualization_request['type'],
#                'parameters': visualization_request['parameters']
#            }
#
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#def detect_visualization_request(query: str) -> Optional[Dict]:
#    """
#    Analyze the query to detect if a visualization is being requested.
#    Returns visualization type and parameters if detected.
#    """
#    # Simple keyword matching for now - in production, use NLP or LLM for this
#    query_lower = query.lower()
#
#    if any(term in query_lower for term in ["show", "visualize", "map", "display"]):
#        if any(term in query_lower for term in ["electrification", "electricity access"]):
#            return {
#                'type': 'electrification_status',
#                'parameters': {}
            #            }
            #        elif any(term in query_lower for term in ["grid distance", "distance to grid"]):
            #            # Extract distance if mentioned (simple regex approach)
            #            import re
            #            distance_match = re.search(r'(\d+)\s*km', query_lower)
            #            distance = 5000  # Default 5km
            #            if distance_match:
            #                try:
            #                    distance = int(distance_match.group(1)) * 1000  # Convert km to meters
            #                except ValueError:
            #                    pass
            #                    
            #            return {
            #                'type': 'grid_distance',
            #                'parameters': {'buffer_distance': distance}
            #            }
            #        elif any(term in query_lower for term in ["minigrid", "mini-grid", "mini grid"]):
            #            return {
            #                'type': 'minigrid_potential',
            #                'parameters': {}
            #            }
            #        elif any(term in query_lower for term in ["buildings", "density", "population"]):
            #            return {
            #                'type': 'building_density',
            #                'parameters': {}
            #            }
            #        elif any(term in query_lower for term in ["infrastructure", "facilities", "roads"]):
            #            return {
            #                'type': 'infrastructure_analysis',
            #                'parameters': {}
            #            }
            #            
            #    return None

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)