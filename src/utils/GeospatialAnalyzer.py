from typing import List, Dict, Tuple, Optional
import geopandas as gpd
from shapely.geometry import Polygon, Point, base
import rasterio
import numpy as np
# from sqlalchemy import create_engine # Uncomment if using PostGIS
import ee
import folium
from IPython.display import display # Keep for displaying maps in Colab
import warnings

# Ignore specific FutureWarnings from geopandas/shapely
warnings.filterwarnings("ignore", message="Iteration over dataset of unknown size")


# Authentication and Initialization (Keep these at the top)
# Uncomment if you are using Google Earth Engine (GEE)
# ee.Authenticate() # Keep if you use EE
# ee.Initialize(project='ee-isekalala') # Keep if you use EE

# Assuming pip install rasterio and pip install folium has been run

# A note on initialization. The GeospatialAnalyzer class assumes that input geometries to its methods are in the CRS EPSG:4326 (WGS84),
# as is default for leaflet and folium maps. If further iterations are needed, with different input geometries, this may need to be updated
# to handle different CRS inputs and reproject them as needed.
# Please ensure that test regions also follow this convention, or are reprojected accordingly before passing to the methods.
# For operations that require metric calculations (like area), the class will reproject geometries to a metric CRS (default is EPSG:32636, UTM zone 36N for Uganda).


class GeospatialAnalyzer:
    """
    A class to load, manage, and analyze geospatial data for a specific region.
    Contains primitive functions for natural language querying by an LLM.
    """
    def __init__(self,
                 buildings_path: str,
                 minigrids_path: str,
                 tile_stats_path: str,
                 plain_tiles_path: str,
                 database_uri: Optional[str] = None,
                 target_metric_crs: str = "EPSG:32636", # WGS 84 / UTM zone 36N for Uganda
                 target_geographic_crs: str = "EPSG:4326" # WGS84 for visualization
                ):
        """
        Initializes the GeospatialAnalyzer by loading data.

        Args:
            buildings_path: Path to the buildings GeoPackage or Shapefile.
            minigrids_path: Path to the mini-grids GeoJSON or Shapefile.
            tile_stats_path: Path to the tile statistics CSV or GeoJSON.
            plain_tiles_path: Path to the plain tiles GeoJSON or Shapefile.
            database_uri: Optional PostGIS connection URI.
            target_metric_crs: The EPSG code for the preferred metric CRS for calculations.
            target_geographic_crs: The EPSG code for the preferred geographic CRS for visualization.
        """
        self.target_metric_crs = target_metric_crs
        self.target_geographic_crs = target_geographic_crs

        self._buildings_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(buildings_path, ensure_crs=True)
        self._minigrids_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(minigrids_path, ensure_crs=True)
        self._tile_stats_gdf: gpd.GeoDataFrame = self._load_and_process_tile_stats(tile_stats_path)
        self._plain_tiles_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(plain_tiles_path, ensure_crs=True)

        # Merge tile stats with plain tiles for easier spatial queries
        self._joined_tiles_gdf = self._merge_tile_data(self._tile_stats_gdf, self._plain_tiles_gdf)


        # PostGIS setup (uncomment if needed)
        # self._db_engine = create_engine(database_uri) if database_uri else None

        print("Geospatial data loading and initial processing complete.")
        print(f"Buildings CRS: {self._buildings_gdf.crs}")
        print(f"Minigrids CRS: {self._minigrids_gdf.crs}")
        print(f"Plain Tiles CRS: {self._plain_tiles_gdf.crs}") # Assuming all are same after merge
        if not self._joined_tiles_gdf.empty:
            print(f"Joined Tiles CRS: {self._joined_tiles_gdf.crs}")

    def _load_and_validate_gdf(self, path: str, ensure_crs: bool = False) -> gpd.GeoDataFrame:
        """Loads a GeoDataFrame and performs basic validation, optionally setting CRS."""
        try:
            gdf = gpd.read_file(path)
            if gdf.empty:
                print(f"Warning: Loaded GeoDataFrame from {path} is empty.")
            if gdf.crs is None:
                 print(f"Warning: GeoDataFrame from {path} has no CRS.")
                 if ensure_crs:
                     print(f"Assuming and setting CRS to WGS84 ({self.target_geographic_crs}).")
                     gdf = gdf.set_crs(self.target_geographic_crs, allow_override=True) # Default to WGS84
            elif ensure_crs and gdf.crs.to_epsg() != int(self.target_geographic_crs.split(':')[-1]):
                 # Optional: Reproject to a common geographic CRS upon loading
                 print(f"Reprojecting {path} from {gdf.crs} to {self.target_geographic_crs}.")
                 gdf = gdf.to_crs(self.target_geographic_crs)


            # Reproject to the target metric CRS for calculation layers if needed
            # You might want to do this lazily in the primitives or explicitly here for certain layers
            # For now, we'll handle CRS within the primitives where calculations happen.

            return gdf
        except Exception as e:
            print(f"Error loading GeoDataFrame from {path}: {e}")
            return gpd.GeoDataFrame() # Return empty GeoDataFrame on error

    def _load_and_process_tile_stats(self, path: str) -> gpd.GeoDataFrame:
        """Loads and processes the tile statistics data."""
        try:
            pathstring = str(path)
            if pathstring.lower().endswith('.csv'):
                 df = gpd.pd.read_csv(path)
            elif pathstring.lower().endswith('.geojson') or pathstring.lower().endswith('.gpkg'):
                 df = gpd.read_file(path)
            else:
                 raise ValueError(f"Unsupported file format for tile stats: {path}")

            # Ensure 'system:index' or 'id' exists and process columns
            if 'system:index' in df.columns:
                df = df.rename(columns={'system:index': 'id'})
            elif 'id' not in df.columns:
                 # Try to use index as id if no id column
                 print("Warning: Tile stats file has no 'id' or 'system:index' column. Using DataFrame index as 'id'.")
                 df['id'] = df.index.astype(int) # Ensure 'id' is integer

            # Replace empty strings with NaN before filling NaNs with 0
            df = df.replace('', np.nan)

            # Define columns that should be integers vs floats
            int_cols = ['cf_days', 'id'] # Assuming 'id' should be int for merging
            float_cols_to_process = [col for col in df.columns if col not in int_cols + ['geometry']] # Exclude geometry

            for col in int_cols:
                if col in df.columns:
                    try:
                        # Ensure the column is converted to numeric, coercing errors, filling NaNs, and then casting to int.
                        df.loc[:, col] = gpd.pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    except Exception as e:
                        print(f"Warning: Could not convert column '{col}' to int in _load_and_process_tile_stats. Error: {e}")

            for col in float_cols_to_process:
                 if col in df.columns:
                    # Use .loc to avoid SettingWithCopyWarning
                    df.loc[:, col] = df[col].fillna(0).astype(float)


            if 'geometry' not in df.columns and pathstring.lower().endswith('.csv'):
                # If loaded from CSV, geometry will be added during the merge with _plain_tiles_gdf
                 print("Tile stats loaded from CSV. Geometry will be added from plain tiles during merge.")
                 # Create a GeoDataFrame without geometry initially if it was a CSV
                 return gpd.GeoDataFrame(df, geometry=None)

            elif 'geometry' in df.columns and isinstance(df, gpd.GeoDataFrame):
                 print("Tile stats loaded from GeoJSON/GeoPackage. Geometry already present.")
                 # Ensure CRS is set for the tile stats GeoDataFrame if it has geometry
                 if df.crs is None:
                     print(f"Warning: Tile stats GeoDataFrame has no CRS. Assuming and setting to WGS84 ({self.target_geographic_crs}).")
                     df = df.set_crs(self.target_geographic_crs, allow_override=True)
                 return df

            else:
                 raise TypeError("Tile stats data could not be processed into a valid GeoDataFrame with or without geometry.")


        except Exception as e:
            print(f"Error loading or processing tile stats from {path}: {e}")
            # Return an empty GeoDataFrame with expected columns to prevent errors later
            return gpd.GeoDataFrame(columns=['id', 'cf_days', 'ndvi_mean', 'geometry'], geometry='geometry')


    def _merge_tile_data(self, tile_stats_gdf: gpd.GeoDataFrame, plain_tiles_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Merges tile statistics with plain tile geometries."""
        if tile_stats_gdf.empty or plain_tiles_gdf.empty:
            print("Warning: Cannot merge tile data because one or both GeoDataFrames are empty.")
            return gpd.GeoDataFrame(columns=['id', 'cf_days', 'ndvi_mean', 'geometry'], geometry='geometry')

        if 'id' not in tile_stats_gdf.columns:
             print("Error: Tile stats GeoDataFrame is missing the 'id' column for merging.")
             return gpd.GeoDataFrame(columns=['id', 'cf_days', 'ndvi_mean', 'geometry'], geometry='geometry')
        if 'id' not in plain_tiles_gdf.columns:
             # Create an 'id' column in plain_tiles_gdf from its index if missing
             print("Warning: Plain tiles GeoDataFrame is missing the 'id' column. Creating from index.")
             plain_tiles_gdf['id'] = plain_tiles_gdf.index.astype(int)


        print("Merging tile stats and plain tiles on 'id'...")
        # Merge the dataframes on the 'id' column
        merged_gdf = tile_stats_gdf.merge(plain_tiles_gdf[['id', 'geometry']], on='id', how='left')

        # Ensure the result is a GeoDataFrame and has a geometry column
        if not isinstance(merged_gdf, gpd.GeoDataFrame):
             merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry')

        if 'geometry' not in merged_gdf.columns or merged_gdf['geometry'].isnull().all():
             print("Error: Merge resulted in a GeoDataFrame without valid geometry.")
             return gpd.GeoDataFrame(columns=['id', 'cf_days', 'ndvi_mean', 'geometry'], geometry='geometry')

        # Ensure merged GeoDataFrame has a CRS, ideally the same as the plain tiles
        if merged_gdf.crs is None and plain_tiles_gdf.crs is not None:
             print(f"Setting merged tiles CRS to match plain tiles CRS: {plain_tiles_gdf.crs}")
             merged_gdf = merged_gdf.set_crs(plain_tiles_gdf.crs)
        elif merged_gdf.crs is None:
             print(f"Warning: Merged tiles GeoDataFrame has no CRS. Setting to WGS84 ({self.target_geographic_crs}).")
             merged_gdf = merged_gdf.set_crs(self.target_geographic_crs, allow_override=True)

        print("Merge complete.")
        return merged_gdf


    # Helper to ensure a geometry has a CRS for calculations
    def _prepare_geometry_for_crs(self, geometry: base.BaseGeometry, target_crs: str) -> Tuple[gpd.GeoSeries, bool]:
        """
        Ensures a Shapely geometry is in the target CRS for calculations.
        Returns the reprojected geometry and a boolean indicating if reprojection occurred.
        """
        geom_series = gpd.GeoSeries([geometry], crs=self.target_geographic_crs)
        reprojected = False

        if self.target_geographic_crs != target_crs:
            geom_series = geom_series.to_crs(target_crs)
            reprojected = True

        return geom_series, reprojected


    # Helper to ensure a GeoDataFrame has a CRS for calculations
    def _check_and_reproject_gdf(self, gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
         """
         Ensures a GeoDataFrame is in the target CRS for calculations.
         Returns the reprojected GeoDataFrame.
         """
         if gdf.crs is None:
              print(f"Warning: GeoDataFrame for calculation has no CRS. Assuming {self.target_geographic_crs}.")
              # Assume a CRS if none exists, then proceed. This is a fallback.
              gdf = gdf.set_crs(self.target_geographic_crs, allow_override=True)

         if gdf.crs.to_epsg() != int(target_crs.split(':')[-1]):
              print(f"Reprojecting GeoDataFrame from {gdf.crs} to {target_crs} for calculation.")
              gdf = gdf.to_crs(target_crs)
         return gdf
    # -----------------------------------------------------------------------------
    # 0) Tester primitives
    # -----------------------------------------------------------------------------


    def get_tile_ids_within_region(self, region: Polygon) -> List[int]:
        """
        Returns the IDs of tiles whose geometry intersects the given region.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            A list of tile IDs that intersect with the region.
        """
        if self._joined_tiles_gdf.empty or 'id' not in self._joined_tiles_gdf.columns:
            print("Error: Joined tiles data is empty or missing 'id' column for get_tile_ids_within.")
            return []

        gdf = self._joined_tiles_gdf.copy()  # Work on a copy to avoid modifying original data

        # Ensure consistent CRS for tile intersection with the region
        region_for_intersect, _ = self._prepare_geometry_for_crs(region, gdf.crs)

        try:
            intersecting_tiles = gdf.loc[gdf.intersects(region_for_intersect)]
            return intersecting_tiles['id'].tolist()
        except Exception as e:
            print(f"Error finding tile IDs within region: {e}")
            return []
    def get_gdf_info_within_region(self,
        region: Polygon,
        layer_name: str, # Use layer name instead of gdf directly
        filter_expr: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Returns a GeoDataFrame of features whose geometry intersects the given region.
        Optionally applies a pandas-style filter expression first.

        Args:
            region: The Shapely Polygon defining the area of interest.
            layer_name: The name of the layer to query ('buildings', 'minigrids', 'tiles').
            filter_expr: Optional pandas-style query string (e.g., "type=='residential'").

        Returns:
            A GeoDataFrame containing the intersecting features and their attributes.
        """
        layer_map = {
            'buildings': self._buildings_gdf,
            'minigrids': self._minigrids_gdf,
            'tiles': self._joined_tiles_gdf # Use the joined gdf for tile queries
        }
        if layer_name not in layer_map:
            print(f"Error: Unknown layer name '{layer_name}'. Available layers: {list(layer_map.keys())}")
            return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')

        gdf = layer_map[layer_name].copy()
        # Ensure consistent CRS for intersection
        region_for_intersect, _ = self._prepare_geometry_for_crs(region, gdf.crs)
        region_geom = region_for_intersect.geometry.iloc[0]
        try:
            if filter_expr:
                gdf = gdf.query(filter_expr)
            intersecting_features = gdf.loc[gdf.intersects(region_geom)]
            return intersecting_features
        except Exception as e:
            print(f"Error finding features within region: {e}")
            return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')
        
        
    def get_tiles_info_within_region(self, region: Polygon) -> gpd.GeoDataFrame:
        """
        Returns a GeoDataFrame of tiles whose geometry intersects the given region.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            A GeoDataFrame containing the intersecting tiles and their attributes.
        """
        return self.get_gdf_info_within_region(region, 'tiles')
        
    def get_minigrids_info_within_region(self, region: Polygon) -> gpd.GeoDataFrame:
        """
        Returns a GeoDataFrame of mini-grids whose geometry intersects the given region.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            A GeoDataFrame containing the intersecting mini-grids and their attributes.
        """
        return self.get_gdf_info_within_region(region, 'minigrids')
        
    # -----------------------------------------------------------------------------
    # 1) Generic vector‐counting primitive
    # -----------------------------------------------------------------------------
    def count_features_within_region(self,
        region: Polygon,
        layer_name: str, # Use layer name instead of gdf directly
        filter_expr: Optional[str] = None
    ) -> int:
        """
        Counts features in a specified layer whose geometry intersects the region.
        Optionally applies a pandas-style filter expression first.

        Args:
            region: The Shapely Polygon defining the area of interest.
            layer_name: The name of the layer to count features from ('buildings', 'minigrids', 'tiles').
            filter_expr: Optional pandas-style query string (e.g., "type=='residential'").

        Returns:
            The number of intersecting features.
        """
        layer_map = {
            'buildings': self._buildings_gdf,
            'minigrids': self._minigrids_gdf,
            'tiles': self._joined_tiles_gdf # Use the joined gdf for tile queries
        }
        if layer_name not in layer_map:
            print(f"Error: Unknown layer name '{layer_name}'. Available layers: {list(layer_map.keys())}")
            return 0

        gdf = layer_map[layer_name].copy() # Work on a copy to avoid modifying original data

        if gdf.empty:
             print(f"Warning: Layer '{layer_name}' is empty. Count is 0.")
             return 0

        if filter_expr:
            try:
                gdf = gdf.query(filter_expr)
            except Exception as e:
                print(f"Error applying filter expression '{filter_expr}' to layer '{layer_name}': {e}")
                return 0 # Return 0 if filter fails

        if gdf.empty:
             print(f"Warning: Layer '{layer_name}' is empty after filtering. Count is 0.")
             return 0

        # Ensure consistent CRS for intersection
        region_for_intersect, _ = self._prepare_geometry_for_crs(region, gdf.crs)


        try:
            # Use .loc to avoid SettingWithCopyWarning
            clipped = gdf.loc[gdf.intersects(region_for_intersect.geometry.iloc[0])]
            return len(clipped)
        except Exception as e:
            print(f"Error during intersection for layer '{layer_name}': {e}")
            return 0

    # -----------------------------------------------------------------------------
    # 2) Building‐specific counts
    # -----------------------------------------------------------------------------
    def count_buildings_within_region(self, region: Polygon) -> int:
        """
        Counts all building footprints within the region.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The number of buildings within the region.
        """
        return self.count_features_within_region(region, 'buildings')

    def count_high_ndvi_buildings(self,
        region: Polygon,
        ndvi_threshold: float = 0.4
    ) -> int:
        """
        Counts buildings whose intersected tile-based NDVI_mean > threshold.

        Args:
            region: The Shapely Polygon defining the area of interest.
            ndvi_threshold: The minimum average NDVI for a tile to be considered 'high'.

        Returns:
            The number of buildings within high-NDVI tile areas within the region.
        """
        # Use the joined tiles gdf which includes both geometry and ndvi_mean
        if self._joined_tiles_gdf.empty or 'ndvi_mean' not in self._joined_tiles_gdf.columns:
             print("Error: Joined tiles data is empty or missing 'ndvi_mean' for count_high_ndvi_buildings.")
             return 0

        # Ensure consistent CRS for tile intersection with the region
        tiles_for_intersect = self._check_and_reproject_gdf(self._joined_tiles_gdf.copy(), region.crs)
        region_for_tiles_intersect = region # Assuming region's CRS is the target

        tiles_in_region = tiles_for_intersect.loc[tiles_for_intersect.intersects(region_for_tiles_intersect)].copy()

        if tiles_in_region.empty:
             return 0

        # Keep only high-NDVI tiles
        high_ndvi_tiles = tiles_in_region.loc[tiles_in_region["ndvi_mean"] > ndvi_threshold].copy()

        if high_ndvi_tiles.empty:
             return 0

        # Buffer those tiles into a unioned polygon
        # Ensure metric CRS for accurate buffering and union
        high_ndvi_tiles_metric = self._check_and_reproject_gdf(high_ndvi_tiles, self.target_metric_crs)


        try:
             highveg_area_metric = high_ndvi_tiles_metric.unary_union
        except Exception as e:
             print(f"Error performing unary_union on high NDVI tiles for count_high_ndvi_buildings: {e}")
             return 0

        # Intersect buildings with that highveg_area ∩ region
        # Ensure buildings_gdf is in the same CRS as the highveg_area_metric for intersection
        buildings_to_intersect = self._check_and_reproject_gdf(self._buildings_gdf.copy(), highveg_area_metric.crs)

        # Ensure the region is also in the metric CRS for the final intersection
        region_metric, _ = self._prepare_geometry_for_crs(region, self.target_metric_crs)


        try:
            # Intersect buildings with the high vegetation area and the region
            # Note: This can be computationally intensive for large datasets
            intersected_buildings = buildings_to_intersect.loc[
                 buildings_to_intersect.intersects(highveg_area_metric) &
                 buildings_to_intersect.intersects(region_metric)
            ]
            return len(intersected_buildings)
        except Exception as e:
             print(f"Error during building intersection with high vegetation area and region in count_high_ndvi_buildings: {e}")
             return 0


    # -----------------------------------------------------------------------------
    # 3) NDVI & other tile‐based stats
    # -----------------------------------------------------------------------------
    def avg_tile_stat(self, region: Polygon, stat: str) -> float:
        """
        Calculates the area-weighted average statistic for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted average statistic, or NaN if no tiles intersect or total area is zero.
        """
        # Use the joined tiles gdf
        if self._joined_tiles_gdf.empty or stat not in self._joined_tiles_gdf.columns:
             print("Error: Joined tiles data is empty or missing {stat}.")
             return float("nan")

        # Ensure consistent CRS for tile intersection with the region
        gdf = self._joined_tiles_gdf.copy() # Work on a copy to avoid modifying original data 

        tiles_m = self._check_and_reproject_gdf(gdf, self.target_metric_crs)
        region_m, _ = self._prepare_geometry_for_crs(region, self.target_metric_crs)
        region_m_geom = region_m.geometry.iloc[0]

        tiles = tiles_m.loc[tiles_m.intersects(region_m_geom)]

        if tiles.empty:
            return float("nan")
        
        try:
            # Ensure intersection is done in a projected CRS for area calculation
            tiles = tiles.copy()  # Avoid SettingWithCopyWarning
            tiles["intersect_area"] = tiles.geometry.intersection(region_m_geom).area
            weighted = (tiles[stat] * tiles["intersect_area"]).sum()
            total   = tiles["intersect_area"].sum()
            return weighted / total if total > 0 else float("nan")
        except Exception as e:
            print(f"Error calculating area-weighted average {stat}: {e}")
            return float("nan")

    def avg_ndvi(self, region: Polygon) -> float:
        """
        Calculates the area-weighted average NDVI for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted average NDVI, or NaN if no tiles intersect or total area is zero.
        """
        return self.avg_tile_stat(region, 'ndvi_mean')
    def cf_days(self, region: Polygon) -> float:
        """
        Calculates the mean total cloud-free days for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The mean total cloud-free days, or NaN if no tiles intersect or total area is zero.
        """
        return self.avg_tile_stat(region, 'cf_days')
    def evi_med(self, region: Polygon) -> float:
        """
        Calculates the area-weighted median EVI for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted median EVI, or NaN if no tiles intersect or total area is zero.
        """
        return self.avg_tile_stat(region, 'evi_med')
    def elev_mean(self, region: Polygon) -> float:
        """
        Calculates the area-weighted mean elevation for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted mean elevation, or NaN if no tiles intersect or total area is zero.
        """
        return self.avg_tile_stat(region, 'elev_mean')
    def slope_mean(self, region: Polygon) -> float:
        """
        Calculates the area-weighted mean slope for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted mean slope, or NaN if no tiles intersect or total area is zero.
        """
        return self.avg_tile_stat(region, 'slope_mean')
    def par_mean(self, region: Polygon) -> float:
        """
        Calculates the area-weighted mean PAR (Photosynthetically Active Radiation) for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted mean PAR, or NaN if no tiles intersect or total area is zero.
        """
        return self.avg_tile_stat(region, 'par_mean')
    

    def ndvi_stats(self, region: Polygon) -> Dict[str, float]:
        """
        Calculates descriptive statistics (mean, median, std) for NDVI
        of the tiles overlapping the region.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            A dictionary containing NDVI statistics, or a dictionary with NaN values
            if no tiles overlap or required columns are missing.
        """
        # Use the joined tiles gdf and match CSV columns
        if self._joined_tiles_gdf.empty or not {'ndvi_mean', 'ndvi_med', 'ndvi_std'}.issubset(self._joined_tiles_gdf.columns):
            print("Error: Joined tiles data is empty or missing required NDVI columns for ndvi_stats.")
            return {
                "NDVI_mean": float("nan"),
                "NDVI_med": float("nan"),
                "NDVI_std": float("nan"),
            }

        # Ensure consistent CRS for tile intersection with the region
        mean = self.avg_ndvi(region)
        median = self.avg_tile_stat(region, 'ndvi_med')
        std = self.avg_tile_stat(region, 'ndvi_std')
        
        return {
            "NDVI_mean": (mean),
            "NDVI_med": (median),
            "NDVI_std": (std),
        }

    # -----------------------------------------------------------------------------
    # 4) Nearest‐neighbor queries on mini‐grids
    # -----------------------------------------------------------------------------
    def list_mini_grids(self) -> List[str]:
        """
        Returns the site names or IDs of all mini-grid locations.

        Returns:
            A list of mini-grid site IDs.
        """
        if self._minigrids_gdf.empty:
             print("Warning: No mini-grid data loaded.")
             return []
        if 'Location' not in self._minigrids_gdf.columns:
             print("Warning: 'Location' column not found in mini-grids data. Returning index.")
             return self._minigrids_gdf.index.astype(str).tolist()
        return self._minigrids_gdf["Location"].tolist()

    def get_site_geometry(self, site_id: str) -> Optional[Polygon]:
        """
        Returns the Shapely geometry for a given mini-grid site_id.

        Args:
            site_id: The ID of the mini-grid site.

        Returns:
            The Shapely Polygon geometry, or None if the site_id is not found.
        """
        if self._minigrids_gdf.empty or 'site_id' not in self._minigrids_gdf.columns:
             print("Error: Mini-grid data is empty or missing 'site_id' for get_site_geometry.")
             return None
        row = self._minigrids_gdf[self._minigrids_gdf["site_id"] == site_id]
        if not row.empty:
            return row.geometry.values[0]
        else:
            print(f"Warning: Mini-grid site ID '{site_id}' not found.")
            return None

    def nearest_mini_grids(self,
        pt: Point,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Finds the k closest mini-grid sites to a given point.

        Args:
            pt: The Shapely Point for the query location.
            k: The number of nearest mini-grids to return.

        Returns:
            A list of tuples (site_id, distance_meters). Returns an empty list
            if no mini-grids are available or an error occurs.
        """
        if self._minigrids_gdf.empty:
             print("Warning: No mini-grid data loaded for nearest_mini_grids.")
             return []

        # Ensure minigrids GeoDataFrame is in a metric CRS for accurate distance calculation
        minigrids_metric = self._check_and_reproject_gdf(self._minigrids_gdf.copy(), self.target_metric_crs)

        # Ensure the query point is also in the same metric CRS
        point_metric, _ = self._prepare_geometry_for_crs(pt, self.target_metric_crs)


        try:
            minigrids_metric.loc[:, "distance"] = minigrids_metric.geometry.distance(point_metric)
            nearest = minigrids_metric.nsmallest(k, "distance")

            if nearest.empty:
                 return []

            if 'site_id' not in nearest.columns:
                 print("Warning: 'site_id' column not found for nearest mini-grids. Returning index.")
                 return list(zip(nearest.index.astype(str), nearest["distance"]))

            return list(zip(nearest["site_id"], nearest["distance"]))
        except Exception as e:
             print(f"Error finding nearest mini-grids: {e}")
             return []


    # -----------------------------------------------------------------------------
    # 5) Generic SQL‐backed primitive (PostGIS) - Uncomment if using
    # -----------------------------------------------------------------------------
    # def query_postgis(self, sql: str) -> gpd.GeoDataFrame:
    #     """
    #     Runs a raw SQL query against PostGIS and returns a GeoDataFrame.
    #
    #     Args:
    #         sql: The SQL query string.
    #
    #     Returns:
    #         A GeoDataFrame containing the query results.
    #     """
    #     if self._db_engine is None:
    #          print("Error: PostGIS database engine not initialized.")
    #          return gpd.GeoDataFrame()
    #     try:
    #         return gpd.read_postgis(sql, self._db_engine, geom_col="geom")
    #     except Exception as e:
    #          print(f"Error executing PostGIS query: {e}")
    #          return gpd.GeoDataFrame()

    # def avg_ndvi_postgis(self, region: Polygon) -> float:
    #     """
    #     Computes area‐weighted average NDVI via PostGIS SQL.
    #
    #     Args:
    #         region: The Shapely Polygon defining the area of interest.
    #
    #     Returns:
    #         The area-weighted average NDVI from PostGIS, or NaN if an error occurs.
    #     """
    #     if self._db_engine is None:
    #          print("Error: PostGIS database engine not initialized.")
    #          return float("nan")
    #     try:
    #         # Ensure region is in a suitable CRS for PostGIS (assuming 4326 for WKT)
    #         region_4326, _ = self._ensure_crs_for_calculation(region, "EPSG:4326")
    #         wkt = region_4326.wkt
    #         # Assuming your tile_stats table in PostGIS has columns ndvi_mean and geom (with SRID 4326)
    #         sql = f"""
    #         SELECT SUM(t.ndvi_mean * ST_Area(ST_Intersection(t.geom, ST_GeomFromText('{wkt}', 4326))))
    #                / SUM(ST_Area(ST_Intersection(t.geom, ST_GeomFromText('{wkt}', 4326))))
    #           AS avg_ndvi
    #         FROM tile_stats t
    #         WHERE ST_Intersects(t.geom, ST_GeomFromText('{wkt}', 4326));
    #         """
    #         df = self.query_postgis(sql)
    #         if not df.empty and 'avg_ndvi' in df.columns:
    #             return float(df["avg_ndvi"].iloc[0])
    #         else:
    #             print("Warning: PostGIS query returned no results or expected column for avg_ndvi_postgis.")
    #             return float("nan")
    #     except Exception as e:
    #          print(f"Error executing PostGIS average NDVI query: {e}")
    #          return float("nan")


    # -----------------------------------------------------------------------------
    # 6) Raster‐on‐the‐fly via Earth Engine - Uncomment if using
    # -----------------------------------------------------------------------------
    # def compute_ndvi_ee(self,
    #     region: base.BaseGeometry,
    #     year: int = 2024
    # ) -> float:
    #     """
    #     Computes area-weighted mean NDVI for region in GEE using Sentinel-2 SR.
    #
    #     Args:
    #         region: The Shapely Geometry defining the area of interest.
    #         year: The year for filtering Sentinel-2 imagery.
    #
    #     Returns:
    #         The computed mean NDVI from Earth Engine, or NaN if an error occurs.
    #     """
    #     if not ee.data.is_initialized():
    #          print("Error: Earth Engine not initialized. Please run ee.Authenticate() and ee.Initialize().")
    #          return float("nan")
    #
    #     try:
    #         # Convert Shapely to EE geometry. Ensure region is in a geographic CRS for GEE.
    #         region_geographic, _ = self._ensure_crs_for_calculation(region, self.target_geographic_crs)
    #
    #         # Handle different geometry types if necessary (Point, LineString, etc.)
    #         if isinstance(region_geographic, Polygon):
    #              ee_geom = ee.Geometry.Polygon(region_geographic.exterior.coords)
    #         elif isinstance(region_geographic, Point):
    #              ee_geom = ee.Geometry.Point(region_geographic.x, region_geographic.y)
    #         else:
    #              print(f"Error: Unsupported geometry type for GEE: {type(region_geographic)}")
    #              return float("nan")
    #
    #         coll = (ee.ImageCollection("COPERNICUS/S2_SR")
    #                 .filterBounds(ee_geom)
    #                 .filterDate(f"{year}-01-01", f"{year}-12-31")
    #                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
    #
    #         if coll.size().getInfo() == 0:
    #              print(f"Warning: No Sentinel-2 imagery found for the region in {year}.")
    #              return float("nan")
    #
    #         ndvi = coll.map(lambda img: img.normalizedDifference(["B8","B4"]).rename("NDVI")).median()
    #
    #         # Check if ndvi image is valid before reducing
    #         try:
    #              info = ndvi.getInfo()
    #              if info is None:
    #                   print("Error: Could not get info for NDVI image collection in GEE.")
    #                   return float("nan")
    #         except Exception as info_e:
    #              print(f"Error getting NDVI image info from GEE: {info_e}")
    #              return float("nan")
    #
    #         stat = ndvi.reduceRegion(
    #             reducer=ee.Reducer.mean(),
    #             geometry=ee_geom,
    #             scale=30, # Sentinel-2 resolution
    #             bestEffort=True, # Use bestEffort for potentially large regions
    #             maxPixels=1e9 # Increase maxPixels if needed
    #         ).get("NDVI")
    #
    #         stat_value = stat.getInfo()
    #         return float(stat_value)
    #     except Exception as e:
    #          print(f"Error computing NDVI in Earth Engine: {e}")
    #          return float("nan")


    # -----------------------------------------------------------------------------
    # 7) Buffer & intersect utility
    # -----------------------------------------------------------------------------
    def buffer_geometry(self,
        geom: base.BaseGeometry,
        radius_m: float
    ) -> base.BaseGeometry:
        """
        Buffers a Shapely geometry by radius_m meters.

        Args:
            geom: The Shapely Geometry to buffer.
            radius_m: The buffer distance in meters.

        Returns:
            The buffered Shapely Geometry.
        """
        # To buffer in meters, the geometry needs to be in a metric CRS.
        # Assuming input geodataframes were loaded with appropriate CRS.
        # It's best practice to reproject if the input geometry is in geographic CRS (like 4326).

        # Check the CRS of the input geometry. If it's geographic, reproject.
        # This requires the input geometry to have a CRS attribute.
        # Shapely geometries don't inherently have CRS. GeoPandas GeoSeries/GeoDataFrame do.
        # If the input `geom` is a direct Shapely object without a CRS, you need context.
        # A safer approach is to assume the input is from one of the loaded geodataframes
        # which are handled during initialization.

        # For demonstration, let's assume the input geometry is associated with a CRS.
        # If not, you might need to pass the source GeoDataFrame's CRS or assume one.
        # This is a simplification for the example.

        original_crs = None
        if hasattr(geom, 'crs') and geom.crs is not None:
             original_crs = geom.crs
        elif hasattr(geom, 'index') and isinstance(geom.index, gpd.GeoSeries): # Check if it's a GeoSeries
             original_crs = geom.index.crs
        # Add other checks for how the geometry is represented and if it has CRS info

        geom_to_buffer = geom

        # If in geographic CRS (like WGS84), reproject to a suitable metric CRS (like UTM)
        # Need to know the appropriate UTM zone for the AOI (Lamwo, Uganda is around 36N)
        geographic_crs_codes = [4326] # WGS84


        if original_crs and original_crs.to_epsg() in geographic_crs_codes:
             print(f"Reprojecting geometry from {original_crs} to {self.target_metric_crs} for buffering.")
             # Create a temporary GeoSeries to reproject the Shapely geometry
             temp_gs = gpd.GeoSeries([geom], crs=original_crs).to_crs(self.target_metric_crs)
             geom_to_buffer = temp_gs.iloc[0]
             buffered_geom = geom_to_buffer.buffer(radius_m)
             # Optional: Reproject back to the original CRS if needed
             # buffered_geom = gpd.GeoSeries([buffered_geom], crs=self.target_metric_crs).to_crs(original_crs).iloc[0]
             return buffered_geom
        else:
             # Assume the geometry is already in a suitable metric CRS
             # Or if no CRS info, perform buffer directly (less accurate)
             print("Warning: Input geometry for buffering has no or non-geographic CRS. Buffering directly (accuracy depends on input CRS).")
             return geom_to_buffer.buffer(radius_m)


    # -----------------------------------------------------------------------------
    # 8) Visualization Primitive (for verification/output)
    # -----------------------------------------------------------------------------
    def visualize_layers(self,
                         center_point: Optional[Point] = None,
                         zoom_start: int = 12,
                         show_buildings: bool = False,
                         show_minigrids: bool = True,
                         show_tiles: bool = False,
                         show_tile_stats: bool = False # Option to style tiles based on stats
                        ) -> folium.Map:
        """
        Visualizes selected layers on a Folium map.

        Args:
            center_point: A Shapely Point to center the map. If None, uses the centroid of the first tile.
            zoom_start: The initial zoom level of the map.
            show_buildings: Whether to add the buildings layer. (Can be slow for many features).
            show_minigrids: Whether to add the mini-grids layer.
            show_tiles: Whether to add the plain tiles layer.
            show_tile_stats: Whether to add the tile stats layer (styled by NDVI).

        Returns:
            A Folium Map object.
        """
        map_center = [0, 0] # Default center

        if center_point is None:
            if not self._plain_tiles_gdf.empty:
                # Ensure the centroid calculation handles CRS
                try:
                    # Reproject to a suitable geographic CRS for Folium
                    tiles_for_centroid = self._check_and_reproject_gdf(self._plain_tiles_gdf.copy(), self.target_geographic_crs)
                    calculated_center = tiles_for_centroid.geometry.centroid.iloc[0]
                    map_center = [calculated_center.y, calculated_center.x]
                except Exception as e:
                    print(f"Warning: Could not calculate map center from plain tiles: {e}. Using default center.")
            else:
                 print("Warning: Plain tiles data is empty. Cannot calculate map center. Using default center.")
                 map_center = [0, 0] # Use a global default if no data

        else:
            # Ensure the input center point is in a geographic CRS for Folium
            try:
                center_point_geographic, _ = self._prepare_geometry_for_crs(center_point, self.target_geographic_crs)
                map_center = [center_point_geographic.y, center_point_geographic.x]
            except Exception as e:
                print(f"Warning: Could not use provided center point due to CRS issues: {e}. Using default center.")
                map_center = [0, 0]


        # Create a base Folium map
        m = folium.Map(location=map_center, zoom_start=zoom_start)

        # Add layers based on parameters
        if show_tiles and not self._plain_tiles_gdf.empty:
            try:
                # Reproject to geographic CRS for Folium
                tiles_for_vis = self._check_and_reproject_gdf(self._plain_tiles_gdf.copy(), self.target_geographic_crs)
                folium.GeoJson(
                    tiles_for_vis.to_json(),
                    name='Plain Tiles',
                    style_function=lambda feature: {'fillColor': 'none', 'color': 'gray', 'weight': 1}
                ).add_to(m)
            except Exception as e:
                print(f"Error adding plain tiles to map: {e}")


        if show_minigrids and not self._minigrids_gdf.empty:
            try:
                # Reproject to geographic CRS for Folium
                minigrids_for_vis = self._check_and_reproject_gdf(self._minigrids_gdf.copy(), self.target_geographic_crs)
                folium.GeoJson(
                    minigrids_for_vis.to_json(),
                    name='Mini Grids',
                    marker=folium.CircleMarker(radius=5, fill=True, fill_color='red', color='red')
                ).add_to(m)
            except Exception as e:
                 print(f"Error adding mini grids to map: {e}")


        if show_buildings and not self._buildings_gdf.empty:
            # Note: Adding a large number of complex polygons can make the map slow.
            # Consider adding a subset or focusing on buildings within a smaller area if needed.
            try:
                # Reproject to geographic CRS for Folium
                buildings_for_vis = self._check_and_reproject_gdf(self._buildings_gdf.copy(), self.target_geographic_crs)
                # Limit the number of buildings for performance if necessary
                # buildings_for_vis = buildings_for_vis.head(1000)
                folium.GeoJson(
                    buildings_for_vis.to_json(),
                    name='Buildings',
                    style_function=lambda feature: {
                        'fillColor': 'blue',
                        'color': 'blue',
                        'weight': 1,
                        'fillOpacity': 0.2
                    }
                ).add_to(m)
            except Exception as e:
                print(f"Error adding buildings to map: {e}")


        if show_tile_stats and not self._joined_tiles_gdf.empty and 'ndvi_mean' in self._joined_tiles_gdf.columns:
             try:
                 # Reproject to geographic CRS for Folium
                 tiles_stats_for_vis = self._check_and_reproject_gdf(self._joined_tiles_gdf.copy(), self.target_geographic_crs)
                 folium.GeoJson(
                     tiles_stats_for_vis.to_json(),
                     name='Tile Stats (NDVI)',
                     style_function=lambda feature: {
                         'fillColor': 'green' if feature['properties'].get('ndvi_mean', 0) > 0.4 else 'orange',
                         'color': 'green' if feature['properties'].get('ndvi_mean', 0) > 0.4 else 'orange',
                         'weight': 1,
                         'fillOpacity': 0.5
                     },
                     tooltip=folium.features.GeoJsonTooltip(fields=['ndvi_mean'], aliases=['NDVI Mean:'])
                 ).add_to(m)
             except Exception as e:
                  print(f"Error adding tile stats to map: {e}")
        elif show_tile_stats:
             print("Warning: Cannot show tile stats. Joined tiles data is empty or missing 'ndvi_mean' column.")


        # Add layer control to switch layers on/off
        folium.LayerControl().add_to(m)

        # Display the map (this works automatically in a Colab cell)
        # display(m) # Uncomment this line if you need to explicitly display

        return m # Return the map object

