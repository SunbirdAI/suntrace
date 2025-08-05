tools = [
    {
      "type": "function",
      "function": {
        "name": "count_features_within_region",
        "description": "Counts features in a specified geospatial layer that intersect with a given region. Can filter features based on a query expression.",
        "parameters": {
          "type": "object",
          "properties": {
            "region": {
              "type": "string",
              "description": "The area as a Shapely Polygon in WKT format."
            },
            "layer_name": {
              "type": "string",
              "enum": ["buildings", "tiles", "roads", "villages", "parishes",
                "subcounties", "existing_grid", "grid_extension", "candidate_minigrids", "existing_minigrids"]
            },
            "filter_expr": {
              "type": "string",
              "description": "Optional pandas-style filter (e.g. \"type=='residential'\")."
            }
          },
          "required": ["region", "layer_name"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "analyze_region",
        "description": "Performs comprehensive analysis of a geographic region, \
          providing structured insights combining the functions which analyze settlements, infrastructure, and environmental characteristics.",
        "parameters": {
          "type": "object",
          "properties": {
            "region": {
              "type": "string",
              "description": "The geographic area (as a Shapely Polygon in WKT format) to analyze."
            }
          },
          "required": ["region"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "_analyze_environmental_metrics",
        "description": "Performs comprehensive analysis of a geographic region, \
          providing structured insights about environmental characteristics, \
          Example response: \
            {'ndvi_mean': -0.5122, 'ndvi_med': -0.5806,'ndvi_std': 0.2273,'evi_med': 1.4574, \
              'elev_mean': 849.5706, 'slope_mean': 2.5659,'par_mean': 179.2317,'rain_total_mm': 34.5617, \
              'rain_mean_mm_day': 3.2298, 'cloud_free_days': 29.0, 'vegetation_density': 'Very limited vegetation'}",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                  "type": "string",
                  "description": "The geographic area (as a Shapely Polygon in WKT format) to analyze."
                }
            },
          "required": ["region"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "_analyze_settlements_in_region",
        "description": "Analyzes building data and settlement patterns within a specified geographic region. \
          Returns a summary of building counts, categories, and intersecting villages with their electrification categories. \
          Example response: \
          {'building_count': 240, \
          'building_categories': {'residential': 239, 'health facility': 1}, \
          'intersecting_village_count': 3, \
          'intersecting_village_details': [{'name': 'Otaa', 'electrification_category': 'Solar home system'}, \
                                          {'name': 'Mudu Central', 'electrification_category': 'Existing minigrid'}, \
                                          {'name': 'Mudu East', 'electrification_category': 'Candidate minigrid', 'priority_rank': 21}, \
          'has_truncated_villages': False}.",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                  "type": "string",
                  "description": "The geographic area (as a Shapely Polygon in WKT format) to analyze."
                }
            },
          "required": ["region"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "_analyze_infrastructure_in_region",
        "description": "Analyzes infrastructure elements including roads, grid, and energy systems. \
          Example response: \
          {'roads': {'total_road_segments': 32, 'road_types': {'tertiary': 1, 'unclassified': 31}}, \
          'electricity': {'existing_grid_present': False, 'distance_to_existing_grid': 7928.6, 'grid_extension_proposed': False, 'candidate_minigrids_count': 1, \
                          'existing_minigrids_count': 0, 'capacity_distribution': {}, 'population_to_be_served': 568}}",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                  "type": "string",
                  "description": "The geographic area (as a Shapely Polygon in WKT format) to analyze."
                }
            },
          "required": ["region"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "compute_distance_to_grid",
        "description": "Calculates the distance from a given geometry to the nearest grid infrastructure. Returns the distance in meters.",
        "parameters": {
          "type": "object",
          "properties": {
            "geometry": {
              "type": "string",
              "description": "The geometry to measure distance from (as a Shapely Polygon in WKT format)."
            }
          },
          "required": ["geometry"]
        }
      }
    }
]
