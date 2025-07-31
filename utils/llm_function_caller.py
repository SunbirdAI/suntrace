import json

import unicodedata
import os
import openai
from shapely.wkt import loads as wkt_loads
import unicodedata

# Load system prompt from external file
current_dir = os.path.dirname(os.path.abspath(__file__))
system_prompt_file = os.path.abspath(os.path.join(current_dir, '../../configs/system_prompt.txt'))
try:
    with open(system_prompt_file, 'r') as f:
        SYSTEM_PROMPT = f.read()
except Exception as e:
    print(f"Warning: Could not load system prompt: {e}")
    SYSTEM_PROMPT = ""


# ── 1) define tools ──────────────────────────────────────────────────────────
tools = [
    {
        "name": "count_features_within_region",
        "description": "Counts features in a specified geospatial layer that intersect with a given region. Can filter features based on a query expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "The area as a Shapely Polygon in WKT format.",
                },
                "layer_name": {
                    "type": "string",
                    "enum": [
                        "buildings",
                        "tiles",
                        "roads",
                        "villages",
                        "parishes",
                        "subcounties",
                        "existing_grid",
                        "grid_extension",
                        "candidate_minigrids",
                        "existing_minigrids",
                    ],
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Optional pandas-style filter (e.g. \"type=='residential'\").",
                },
            },
            "required": ["region", "layer_name"],
        },
    },
    {
      "name": "analyze_region",
      "description": "Performs comprehensive analysis of a geographic region, \
        providing structured insights about settlements, infrastructure, and environmental characteristics.",
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
    },
    {
      "name": "_analyze_environmental_metrics",
      "description": "Performs comprehensive analysis of a geographic region, \
        providing structured insights about environmental characteristics, \
        including NDVI, EVI, Elevation, Slope, Solar PAR, Rainfall, Cloud Free Days.",
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
    },
    {
      "name": "_analyze_settlements_in_region",
      "description": "Analyzes building data and settlement patterns within a specified geographic region.",
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
    },
    {
      "name": "_analyze_infrastructure_in_region",
      "description": "Analyzes infrastructure elements including roads, grid, and energy systems.",
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
    },
    {
      "name": "get_layer_geometry",
      "description": "Retrieves the Shapely geometry for the union of features of a given layer.",
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
          }
        },
        "required": ["region", "layer_name"]
      }
    }
    
    ]


# ── 3) dispatcher ───────────────────────────────────────────────────────────
def handle_tool_call(tool_name, parameters, geospatial_analyzer=None):
    if geospatial_analyzer is None:
        # Fallback to creating a new instance if needed
        from utils.factory import create_geospatial_analyzer

        geospatial_analyzer = create_geospatial_analyzer()

    try:
        region = wkt_loads(parameters["region"])
        if tool_name == "count_features_within_region":
            return geospatial_analyzer.count_features_within_region(
                region, parameters["layer_name"], parameters.get("filter_expr")
            )
        elif tool_name == "analyze_region":
            region = wkt_loads(parameters["region"])
            return geospatial_analyzer.analyze_region(region)
        elif tool_name == "_analyze_environmental_metrics":
            region = wkt_loads(parameters["region"])
            return geospatial_analyzer._analyze_environmental_metrics(region)
        elif tool_name == "_analyze_settlements_in_region":
            region = wkt_loads(parameters["region"])
            return geospatial_analyzer._analyze_settlements_in_region(region)
        elif tool_name == "_analyze_infrastructure_in_region":
            region = wkt_loads(parameters["region"])
            return geospatial_analyzer._analyze_infrastructure_in_region(region)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    except Exception as e:
        return {"error": str(e)}


# ── 4) helper to run a single user query ─────────────────────────────────────
def ask_with_functions(user_prompt, analyzer=None):
    # Sanitize the user prompt to remove problematic Unicode characters
    user_prompt = (
        unicodedata.normalize("NFKD", user_prompt)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    # Add system message for authoritative style
    system_message = SYSTEM_PROMPT

    # start with just the user message
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    # 1st call: let the model decide if it needs a function
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", messages=messages, functions=tools, function_call="auto"
    )
    msg = response.choices[0].message

    # if it wants to call one of our tools...
    if msg.get("function_call"):
        name = msg.function_call.name
        args = json.loads(msg.function_call.arguments)

        # run the tool
        result = handle_tool_call(name, args, analyzer)

        # add the tool call and its result into the conversation
        messages.append(msg)  # the function_call request
        messages.append(
            {"role": "function", "name": name, "content": json.dumps(result)}
        )

        # 2nd call: let the model produce a final answer
        second = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
        return second.choices[0].message.content

    # otherwise, just return the content
    return msg.content


# ── 5) how to supply a dynamic region ────────────────────────────────────────
# Sample code is commented out to avoid running on import
# e.g. load a polygon from disk (or accept WKT from your front-end)
# polygon_path = os.path.join(current_dir, "../../data/sample_region_mudu/mudu_village.gpkg")
# gdf = gpd.read_file(polygon_path)
# wkt_polygon = gdf.geometry.iloc[0].wkt

# now ask:
# query = (
#     f"How many buildings are within this region? "
#     f"Here's the region WKT: {wkt_polygon}"
# )
# answer = ask_with_functions(query)
# print(answer)

"""  {
    "type": "function",
    "function": {
      "name": "count_high_ndvi_buildings",
      "description": "Counts buildings within a region whose intersected tile-based average NDVI is above a specified threshold.",
      "parameters": {
        "type": "object",
        "properties": {
          "region": {
            "type": "string",
            "description": "The geographic area (as a Shapely Polygon in WKT format) to count high NDVI buildings within."
          },
          "ndvi_threshold": {
            "type": "number",
            "description": "The minimum average NDVI value for a tile to be considered 'high'. Defaults to 0.4."
          }
        },
        "required": ["region"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "avg_ndvi",
      "description": "Calculates the area-weighted average NDVI for a given geographic region using tile statistics.",
      "parameters": {
        "type": "object",
        "properties": {
          "region": {
            "type": "string",
            "description": "The geographic area (as a Shapely Polygon in WKT format) to calculate average NDVI for."
          }
        },
        "required": ["region"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "ndvi_stats",
      "description": "Calculates descriptive statistics (mean, median, standard deviation, minimum, and maximum) for NDVI of the tiles overlapping a given region.",
      "parameters": {
        "type": "object",
        "properties": {
          "region": {
            "type": "string",
            "description": "The geographic area (as a Shapely Polygon in WKT format) to calculate NDVI statistics for."
          }
        },
        "required": ["region"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_mini_grids",
      "description": "Returns a list of the site names or IDs of all mini-grid locations.",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_site_geometry",
      "description": "Retrieves the geographic geometry (as a Polygon) for a specific mini-grid site ID.",
      "parameters": {
        "type": "object",
        "properties": {
          "site_id": {
            "type": "string",
            "description": "The ID of the mini-grid site."
          }
        },
        "required": ["site_id"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "nearest_mini_grids",
      "description": "Finds the k closest mini-grid sites to a given geographic point.",
      "parameters": {
        "type": "object",
        "properties": {
          "pt": {
            "type": "string",
            "description": "The geographic location (as a Shapely Point in WKT format) for the nearest neighbor query."
          },
          "k": {
            "type": "integer",
            "description": "The number of nearest mini-grids to return. Defaults to 3."
          }
        },
        "required": ["pt"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "buffer_geometry",
      "description": "Creates a buffer of a specified radius around a given geographic geometry.",
      "parameters": {
        "type": "object",
        "properties": {
          "geom": {
            "type": "string",
            "description": "The input geographic geometry (as a Shapely geometry in WKT format) to buffer."
          },
          "radius_m": {
            "type": "number",
            "description": "The buffer distance in meters."
          }
        },
        "required": ["geom", "radius_m"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "visualize_layers",
      "description": "Generates an interactive map visualizing selected geospatial layers within the area of interest.",
      "parameters": {
        "type": "object",
        "properties": {
          "center_point": {
            "type": "string",
            "description": "A geographic point (as a Shapely Point in WKT format) to center the map. If not provided, a default center is used."
          },
          "zoom_start": {
            "type": "integer",
            "description": "The initial zoom level of the map. Defaults to 12."
          },
          "show_buildings": {
            "type": "boolean",
            "description": "Whether to include the buildings layer on the map. Note: Can be slow for many features. Defaults to false."
          },
          "show_minigrids": {
            "type": "boolean",
            "description": "Whether to include the mini-grids layer on the map. Defaults to true."
          },
          "show_tiles": {
            "type": "boolean",
            "description": "Whether to include the plain tiles layer on the map. Defaults to false."
          },
          "show_tile_stats": {
            "type": "boolean",
            "description": "Whether to include the tile stats layer (styled by NDVI) on the map. Defaults to false."
          }
        }
      }
    }
  }"""


"""
        elif tool_name == "count_high_ndvi_buildings":
            region = wkt_loads(parameters["region"])
            ndvi_threshold = parameters.get("ndvi_threshold", 0.4)
            return geospatial_analyzer.count_high_ndvi_buildings(region, ndvi_threshold)

        elif tool_name == "avg_ndvi":
            region = wkt_loads(parameters["region"])
            return geospatial_analyzer.avg_ndvi(region)

        elif tool_name == "ndvi_stats":
            region = wkt_loads(parameters["region"])
            return geospatial_analyzer.ndvi_stats(region)

        elif tool_name == "list_mini_grids":
            return geospatial_analyzer.list_mini_grids()

        elif tool_name == "get_site_geometry":
            site_id = parameters["site_id"]
            return geospatial_analyzer.get_site_geometry(site_id)

        elif tool_name == "nearest_mini_grids":
            pt = wkt_loads(parameters["pt"])
            k = parameters.get("k", 3)
            return geospatial_analyzer.nearest_mini_grids(pt, k)

        elif tool_name == "buffer_geometry":
            geom = wkt_loads(parameters["geom"])
            radius_m = parameters["radius_m"]
            return geospatial_analyzer.buffer_geometry(geom, radius_m)

        elif tool_name == "visualize_layers":
            center_point = wkt_loads(parameters["center_point"]) if "center_point" in parameters else None
            zoom_start = parameters.get("zoom_start", 12)
            show_buildings = parameters.get("show_buildings", False)
            show_minigrids = parameters.get("show_minigrids", True)
            show_tiles = parameters.get("show_tiles", False)
            show_tile_stats = parameters.get("show_tile_stats", False)
            return geospatial_analyzer.visualize_layers(
                center_point, zoom_start, show_buildings, show_minigrids, show_tiles, show_tile_stats
            )
"""
