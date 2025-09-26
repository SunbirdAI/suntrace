import json
from typing import List, Dict, Tuple, Optional, Any
import warnings
import geopandas as gpd
import pandas as pd

from shapely.geometry import Polygon, Point, base
# import rasterio
import numpy as np

# from sqlalchemy import create_engine # Uncomment if using PostGIS
# import ee
import folium
# from IPython.display import display  # Keep for displaying maps in Colab


# Ignore specific FutureWarnings from geopandas/shapely
warnings.filterwarnings("ignore", message="Iteration over dataset of unknown size")


# Authentication and Initialization (Keep these at the top)
# Uncomment if you are using Google Earth Engine (GEE)
# ee.Authenticate() # Keep if you use EE
# ee.Initialize(project='project-id') # Keep if you use EE

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

    def __init__(
        self,
        buildings_path: str,
        tile_stats_path: str,
        plain_tiles_path: str,
        candidate_minigrids_path: str,
        existing_minigrids_path: str,
        existing_grid_path: str,
        grid_extension_path: str,
        roads_path: str,
        villages_path: str,
        parishes_path: str,
        subcounties_path: str,
        database_uri: Optional[str] = None,
        target_metric_crs: str = "EPSG:32636",  # WGS 84 / UTM zone 36N for Uganda
        target_geographic_crs: str = "EPSG:4326",  # WGS84 for visualization
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

        self._buildings_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            buildings_path, ensure_crs=True
        )
        self._tile_stats_gdf: gpd.GeoDataFrame = self._load_and_process_tile_stats(
            tile_stats_path
        )
        self._plain_tiles_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            plain_tiles_path, ensure_crs=True
        )
        self._candidate_minigrids_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            candidate_minigrids_path, ensure_crs=True
        )
        self._existing_minigrids_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            existing_minigrids_path, ensure_crs=True
        )
        self._existing_grid_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            existing_grid_path, ensure_crs=True
        )
        self._grid_extension_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            grid_extension_path, ensure_crs=True
        )
        self._roads_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            roads_path, ensure_crs=True
        )
        self._villages_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            villages_path, ensure_crs=True
        )
        self._parishes_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            parishes_path, ensure_crs=True
        )
        self._subcounties_gdf: gpd.GeoDataFrame = self._load_and_validate_gdf(
            subcounties_path, ensure_crs=True
        )

        # Merge tile stats with plain tiles for easier spatial queries
        self._joined_tiles_gdf = self._merge_tile_data(
            self._tile_stats_gdf, self._plain_tiles_gdf
        )

        # PostGIS setup (uncomment if needed)
        # self._db_engine = create_engine(database_uri) if database_uri else None

        print("Geospatial data loading and initial processing complete.")
        print(f"Buildings CRS: {self._buildings_gdf.crs}")
        print(f"Candidate Minigrids CRS: {self._candidate_minigrids_gdf.crs}")
        print(
            f"Plain Tiles CRS: {self._plain_tiles_gdf.crs}"
        )  # Assuming all are same after merge
        if not self._joined_tiles_gdf.empty:
            print(f"Joined Tiles CRS: {self._joined_tiles_gdf.crs}")

        self._layer_map: Dict[str, gpd.GeoDataFrame] = {
            "buildings": self._buildings_gdf,
            "tiles": self._joined_tiles_gdf,
            "tile_stats": self._tile_stats_gdf,
            "roads": self._roads_gdf,
            "villages": self._villages_gdf,
            "parishes": self._parishes_gdf,
            "subcounties": self._subcounties_gdf,
            "existing_grid": self._existing_grid_gdf,
            "grid_extension": self._grid_extension_gdf,
            "candidate_minigrids": self._candidate_minigrids_gdf,
            "existing_minigrids": self._existing_minigrids_gdf,
        }

    def _load_and_validate_gdf(
        self, path: str, ensure_crs: bool = False
    ) -> gpd.GeoDataFrame:
        """Loads a GeoDataFrame and performs basic validation, optionally setting CRS."""
        try:
            gdf = gpd.read_file(path)
            if gdf.empty:
                print(f"Warning: Loaded GeoDataFrame from {path} is empty.")
            if gdf.crs is None:
                print(f"Warning: GeoDataFrame from {path} has no CRS.")
                if ensure_crs:
                    print(
                        f"Assuming and setting CRS to WGS84 ({self.target_geographic_crs})."
                    )
                    gdf = gdf.set_crs(
                        self.target_geographic_crs, allow_override=True
                    )  # Default to WGS84
            elif ensure_crs and gdf.crs.to_epsg() != int(
                self.target_geographic_crs.split(":")[-1]
            ):
                # Optional: Reproject to a common geographic CRS upon loading
                print(
                    f"Reprojecting {path} from {gdf.crs} to {self.target_geographic_crs}."
                )
                gdf = gdf.to_crs(self.target_geographic_crs)

            # Reproject to the target metric CRS for calculation layers if needed
            # You might want to do this lazily in the primitives or explicitly here for certain layers
            # For now, we'll handle CRS within the primitives where calculations happen.

            return gdf
        except Exception as e:
            print(f"Error loading GeoDataFrame from {path}: {e}")
            return gpd.GeoDataFrame()  # Return empty GeoDataFrame on error

    def _load_and_process_tile_stats(self, path: str) -> gpd.GeoDataFrame:
        """Loads and processes the tile statistics data."""
        try:
            pathstring = str(path)
            if pathstring.lower().endswith(".csv"):
                df = gpd.pd.read_csv(path)
            elif pathstring.lower().endswith(".geojson") or pathstring.lower().endswith(
                ".gpkg"
            ):
                df = gpd.read_file(path)
            else:
                raise ValueError(f"Unsupported file format for tile stats: {path}")

            # Ensure 'system:index' or 'id' exists and process columns
            if "system:index" in df.columns:
                df = df.rename(columns={"system:index": "id"})
            elif "id" not in df.columns:
                # Try to use index as id if no id column
                print(
                    "Warning: Tile stats file has no 'id' or 'system:index' column. Using DataFrame index as 'id'."
                )
                df["id"] = df.index.astype(int)  # Ensure 'id' is integer

            # Replace empty strings with NaN before filling NaNs with 0
            df = df.replace("", np.nan)

            # Define columns that should be integers vs floats
            int_cols = [
                "cloud_free_days",
                "id",
            ]  # Assuming 'id' should be int for merging
            float_cols_to_process = [
                col for col in df.columns if col not in int_cols + ["geometry"]
            ]  # Exclude geometry

            for col in int_cols:
                if col in df.columns:
                    try:
                        # Ensure the column is converted to numeric, coercing errors, filling NaNs, and then casting to int.
                        df.loc[:, col] = (
                            gpd.pd.to_numeric(df[col], errors="coerce")
                            .fillna(0)
                            .astype(int)
                        )
                    except Exception as e:
                        print(
                            f"Warning: Could not convert column '{col}' to int in _load_and_process_tile_stats. Error: {e}"
                        )

            for col in float_cols_to_process:
                if col in df.columns:
                    # Use .loc to avoid SettingWithCopyWarning
                    df.loc[:, col] = df[col].fillna(0).astype(float)
            # drop(columns=['bldg_count', 'bldg_area', 'bldg_h_max'])
            df = df.drop(
                columns=["bldg_count", "bldg_area", "bldg_h_max"], errors="ignore"
            )

            if "geometry" not in df.columns and pathstring.lower().endswith(".csv"):
                # If loaded from CSV, geometry will be added during the merge with _plain_tiles_gdf
                print(
                    "Tile stats loaded from CSV. Geometry will be added from plain tiles during merge."
                )
                # Create a GeoDataFrame without geometry initially if it was a CSV
                return gpd.GeoDataFrame(df, geometry=None)

            elif "geometry" in df.columns and isinstance(df, gpd.GeoDataFrame):
                print(
                    "Tile stats loaded from GeoJSON/GeoPackage. Geometry already present."
                )
                # Ensure CRS is set for the tile stats GeoDataFrame if it has geometry
                if df.crs is None:
                    print(
                        f"Warning: Tile stats GeoDataFrame has no CRS. Assuming and setting to WGS84 ({self.target_geographic_crs})."
                    )
                    df = df.set_crs(self.target_geographic_crs, allow_override=True)
                return df

            else:
                raise TypeError(
                    "Tile stats data could not be processed into a valid GeoDataFrame with or without geometry."
                )

        except Exception as e:
            print(f"Error loading or processing tile stats from {path}: {e}")
            # Return an empty GeoDataFrame with expected columns to prevent errors later
            return gpd.GeoDataFrame(
                columns=["id", "cloud_free_days", "ndvi_mean", "geometry"],
                geometry="geometry",
            )

    def _merge_tile_data(
        self, tile_stats_gdf: gpd.GeoDataFrame, plain_tiles_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Merges tile statistics with plain tile geometries."""
        if tile_stats_gdf.empty or plain_tiles_gdf.empty:
            print(
                "Warning: Cannot merge tile data because one or both GeoDataFrames are empty."
            )
            return gpd.GeoDataFrame(
                columns=["id", "cloud_free_days", "ndvi_mean", "geometry"],
                geometry="geometry",
            )

        if "id" not in tile_stats_gdf.columns:
            print(
                "Error: Tile stats GeoDataFrame is missing the 'id' column for merging."
            )
            return gpd.GeoDataFrame(
                columns=["id", "cloud_free_days", "ndvi_mean", "geometry"],
                geometry="geometry",
            )
        if "id" not in plain_tiles_gdf.columns:
            # Create an 'id' column in plain_tiles_gdf from its index if missing
            print(
                "Warning: Plain tiles GeoDataFrame is missing the 'id' column. Creating from index."
            )
            plain_tiles_gdf["id"] = plain_tiles_gdf.index.astype(int)

        print("Merging tile stats and plain tiles on 'id'...")
        # Merge the dataframes on the 'id' column
        merged_gdf = tile_stats_gdf.merge(
            plain_tiles_gdf[["id", "geometry"]], on="id", how="left"
        )

        # Ensure the result is a GeoDataFrame and has a geometry column
        if not isinstance(merged_gdf, gpd.GeoDataFrame):
            merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry="geometry")

        if (
            "geometry" not in merged_gdf.columns
            or merged_gdf["geometry"].isnull().all()
        ):
            print("Error: Merge resulted in a GeoDataFrame without valid geometry.")
            return gpd.GeoDataFrame(
                columns=["id", "cloud_free_days", "ndvi_mean", "geometry"],
                geometry="geometry",
            )

        # Ensure merged GeoDataFrame has a CRS, ideally the same as the plain tiles
        if merged_gdf.crs is None and plain_tiles_gdf.crs is not None:
            print(
                f"Setting merged tiles CRS to match plain tiles CRS: {plain_tiles_gdf.crs}"
            )
            merged_gdf = merged_gdf.set_crs(plain_tiles_gdf.crs)
        elif merged_gdf.crs is None:
            print(
                f"Warning: Merged tiles GeoDataFrame has no CRS. Setting to WGS84 ({self.target_geographic_crs})."
            )
            merged_gdf = merged_gdf.set_crs(
                self.target_geographic_crs, allow_override=True
            )

        print("Merge complete.")
        return merged_gdf

    # Helper to ensure a geometry has a CRS for calculations
    def _prepare_geometry_for_crs(
        self, geometry: base.BaseGeometry, target_crs: str
    ) -> Tuple[gpd.GeoSeries, bool]:
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
    def _check_and_reproject_gdf(
        self, gdf: gpd.GeoDataFrame, target_crs: str
    ) -> gpd.GeoDataFrame:
        """
        Ensures a GeoDataFrame is in the target CRS for calculations.
        Returns the reprojected GeoDataFrame.
        """
        if gdf.crs is None:
            print(
                f"Warning: GeoDataFrame for calculation has no CRS. Assuming {self.target_geographic_crs}."
            )
            # Assume a CRS if none exists, then proceed. This is a fallback.
            gdf = gdf.set_crs(self.target_geographic_crs, allow_override=True)

        if gdf.crs.to_epsg() != int(target_crs.split(":")[-1]):
            print(
                f"Reprojecting GeoDataFrame from {gdf.crs} to {target_crs} for calculation."
            )
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
        if self._joined_tiles_gdf.empty or "id" not in self._joined_tiles_gdf.columns:
            print(
                "Error: Joined tiles data is empty or missing 'id' column for get_tile_ids_within."
            )
            return []

        gdf = (
            self._joined_tiles_gdf.copy()
        )  # Work on a copy to avoid modifying original data

        # Ensure consistent CRS for tile intersection with the region
        region_for_intersect, _ = self._prepare_geometry_for_crs(region, gdf.crs)

        try:
            intersecting_tiles = gdf.loc[gdf.intersects(region_for_intersect)]
            return intersecting_tiles["id"].tolist()
        except Exception as e:
            print(f"Error finding tile IDs within region: {e}")
            return []

    def get_gdf_info_within_region(
        self,
        region: Polygon,
        layer_name: str,  # Use layer name instead of gdf directly
        filter_expr: Optional[str] = None,
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
            "buildings": self._buildings_gdf,
            "tiles": self._joined_tiles_gdf,  # Use the joined gdf for tile queries
            "roads": self._roads_gdf,
            "villages": self._villages_gdf,
            "parishes": self._parishes_gdf,
            "subcounties": self._subcounties_gdf,
            "existing_grid": self._existing_grid_gdf,
            "grid_extension": self._grid_extension_gdf,
            "candidate_minigrids": self._candidate_minigrids_gdf,
            "existing_minigrids": self._existing_minigrids_gdf,
        }
        if layer_name not in layer_map:
            print(
                f"Error: Unknown layer name '{layer_name}'. Available layers: {list(layer_map.keys())}"
            )
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")

        gdf = layer_map[layer_name].copy()
        # Ensure consistent CRS for intersection
        region_for_intersect, _ = self._prepare_geometry_for_crs(region, gdf.crs)
        region_geom = region_for_intersect.geometry.iloc[0]
        try:
            if filter_expr:
                gdf = gdf.query(filter_expr)
            intersecting_features = gdf.loc[gdf.intersects(region_geom)]
            return intersecting_features  # Return as JSON for serialization by LLM
        except Exception as e:
            print(f"Error finding features within region: {e}")
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")

    def get_tiles_info_within_region(self, region: Polygon) -> gpd.GeoDataFrame:
        """
        Returns a GeoDataFrame of tiles whose geometry intersects the given region.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            A GeoDataFrame containing the intersecting tiles and their attributes.
        """
        return self.get_gdf_info_within_region(region, "tiles")

    def get_roads_info_within_region(self, region: Polygon) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, "roads")

    def get_villages_info_within_region(self, region: Polygon) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, "villages")

    def get_parishes_info_within_region(self, region: Polygon) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, "parishes")

    def get_subcounties_info_within_region(self, region: Polygon) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, "subcounties")

    def get_existing_grid_info_within_region(self, region: Polygon) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, "existing_grid")

    def get_grid_extension_info_within_region(
        self, region: Polygon
    ) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, "grid_extension")

    def get_candidate_minigrids_info_within_region(
        self, region: Polygon
    ) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, "candidate_minigrids")

    def get_existing_minigrids_info_within_region(
        self, region: Polygon
    ) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, "existing_minigrids")

    # -----------------------------------------------------------------------------
    # 1) Generic vector‐counting primitive
    # -----------------------------------------------------------------------------
    def count_features_within_region(
        self,
        region: Polygon,
        layer_name: str,  # Use layer name instead of gdf directly
        filter_expr: Optional[str] = None,
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
            "buildings": self._buildings_gdf,
            "tiles": self._joined_tiles_gdf,  # Use the joined gdf for tile queries
            "roads": self._roads_gdf,
            "villages": self._villages_gdf,
            "parishes": self._parishes_gdf,
            "subcounties": self._subcounties_gdf,
            "existing_grid": self._existing_grid_gdf,
            "grid_extension": self._grid_extension_gdf,
            "candidate_minigrids": self._candidate_minigrids_gdf,
            "existing_minigrids": self._existing_minigrids_gdf,
        }
        if layer_name not in layer_map:
            print(
                f"Error: Unknown layer name '{layer_name}'. Available layers: {list(layer_map.keys())}"
            )
            return 0

        gdf = layer_map[
            layer_name
        ].copy()  # Work on a copy to avoid modifying original data

        if gdf.empty:
            print(f"Warning: Layer '{layer_name}' is empty. Count is 0.")
            return 0

        if filter_expr:
            try:
                gdf = gdf.query(filter_expr)
            except Exception as e:
                print(
                    f"Error applying filter expression '{filter_expr}' to layer '{layer_name}': {e}"
                )
                return 0  # Return 0 if filter fails

        if gdf.empty:
            print(
                f"Warning: Layer '{layer_name}' is empty after filtering. Count is 0."
            )
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
    # 3) NDVI & other tile‐based stats
    # -----------------------------------------------------------------------------
    def weighted_tile_stats_all(self, region: Polygon) -> Dict[str, float]:
        """
        Calculates area-weighted averages for all numeric columns in the joined tiles GeoDataFrame
        for the given region.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            A dictionary mapping each numeric column name to its area-weighted average,
            or NaN if no tiles intersect or total area is zero.
        """
        if self._joined_tiles_gdf.empty:
            print("Error: Joined tiles data is empty.")
            return {}

        gdf = self._joined_tiles_gdf.copy()
        tiles_m = self._check_and_reproject_gdf(gdf, self.target_metric_crs)
        region_m, _ = self._prepare_geometry_for_crs(region, self.target_metric_crs)
        region_m_geom = region_m.geometry.iloc[0]

        # Compute region area in metric units (m^2)
        region_area_km2 = float(region_m_geom.area) / 1e6


        tiles = tiles_m.loc[tiles_m.intersects(region_m_geom)]

        if tiles.empty:
            return {}

        try:
            tiles = tiles.copy().drop(
                columns=["id", "tile_total_Mg", "area_m2"], errors="ignore"
            )  # Avoid SettingWithCopyWarning
            tiles["intersect_area"] = tiles.geometry.intersection(region_m_geom).area
            total_area = tiles["intersect_area"].sum()
            if total_area == 0:
                return {}

            # Select numeric columns (excluding geometry and intersect_area)
            numeric_cols = tiles.select_dtypes(include=[np.number]).columns
            numeric_cols = [
                col for col in numeric_cols if col not in ["intersect_area"]
            ]

            weighted_stats = {}
            for col in numeric_cols:
                weighted_sum = (tiles[col] * tiles["intersect_area"]).sum()
                weighted_stats[col] = (
                    weighted_sum / total_area if total_area > 0 else float("nan")
                )
            # include region area in the returned dictionary
            weighted_stats["region_area_km2"] = region_area_km2
            return weighted_stats
        except Exception as e:
            print(f"Error calculating area-weighted averages for all stats: {e}")
            return {}

    def weighted_tile_stat(self, region: Polygon, stat: str) -> float:
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
        gdf = (
            self._joined_tiles_gdf.copy()
        )  # Work on a copy to avoid modifying original data

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
            total = tiles["intersect_area"].sum()
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
        return self.weighted_tile_stat(region, "ndvi_mean")

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
        if self._joined_tiles_gdf.empty or not {
            "ndvi_mean",
            "ndvi_med",
            "ndvi_std",
        }.issubset(self._joined_tiles_gdf.columns):
            print(
                "Error: Joined tiles data is empty or missing required NDVI columns for ndvi_stats."
            )
            return {
                "NDVI_mean": float("nan"),
                "NDVI_med": float("nan"),
                "NDVI_std": float("nan"),
            }

        # Ensure consistent CRS for tile intersection with the region
        mean = self.avg_ndvi(region)
        median = self.weighted_tile_stat(region, "ndvi_med")
        std = self.weighted_tile_stat(region, "ndvi_std")

        return {
            "NDVI_mean": (mean),
            "NDVI_med": (median),
            "NDVI_std": (std),
        }

    def evi_med(self, region: Polygon) -> float:
        """
        Calculates the area-weighted median EVI for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted median EVI, or NaN if no tiles intersect or total area is zero.
        """
        return self.weighted_tile_stat(region, "evi_med")

    def cf_days(self, region: Polygon) -> float:
        """
        Calculates the mean total cloud-free days for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The mean total cloud-free days, or NaN if no tiles intersect or total area is zero.
        """
        return self.weighted_tile_stat(region, "cloud_free_days")

    def elev_mean(self, region: Polygon) -> float:
        """
        Calculates the area-weighted mean elevation for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted mean elevation, or NaN if no tiles intersect or total area is zero.
        """
        return self.weighted_tile_stat(region, "elev_mean")

    def slope_mean(self, region: Polygon) -> float:
        """
        Calculates the area-weighted mean slope for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted mean slope, or NaN if no tiles intersect or total area is zero.
        """
        return self.weighted_tile_stat(region, "slope_mean")

    def par_mean(self, region: Polygon) -> float:
        """
        Calculates the area-weighted mean PAR (Photosynthetically Active Radiation) for a region using tile statistics.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            The area-weighted mean PAR, or NaN if no tiles intersect or total area is zero.
        """
        return self.weighted_tile_stat(region, "par_mean")
    def region_total_biomass(self, region: Polygon, tile_total_col: str = "tile_total_Mg") -> float:
        """
        Returns total biomass (Mg) inside `region` by summing each intersecting tile's
        fractional contribution: tile_total_Mg * (intersection_area / tile_area).
        """
        if self._joined_tiles_gdf.empty or tile_total_col not in self._joined_tiles_gdf.columns:
            print(f"Error: Joined tiles data is empty or missing '{tile_total_col}'.")
            return float("nan")

        # Work on copy and ensure metric CRS for area calculations
        tiles_gdf = self._check_and_reproject_gdf(self._joined_tiles_gdf.copy(), self.target_metric_crs)
        region_m, _ = self._prepare_geometry_for_crs(region, self.target_metric_crs)
        region_geom = region_m.geometry.iloc[0]

        tiles = tiles_gdf.loc[tiles_gdf.intersects(region_geom)].copy()
        if tiles.empty:
            return 0.0

        # Ensure tile area column exists (area_m2)
        if "area_m2" not in tiles.columns:
            tiles["area_m2"] = tiles.geometry.area

        # Compute intersection areas and fractional contributions
        tiles["intersect_area"] = tiles.geometry.intersection(region_geom).area
        # Avoid negative/zero division
        valid = tiles["area_m2"] > 0
        if not valid.any():
            return float("nan")

        tiles.loc[valid, "fraction"] = tiles.loc[valid, "intersect_area"] / tiles.loc[valid, "area_m2"]
        tiles["fraction"] = tiles["fraction"].fillna(0.0).clip(lower=0.0, upper=1.0)

        tiles["contrib_Mg"] = tiles[tile_total_col] * tiles["fraction"]
        return float(tiles["contrib_Mg"].sum())
    # -----------------------------------------------------------------------------
    # 4) Get Layer Geoms and Nearest‐neighbor queries
    # -----------------------------------------------------------------------------
    def list_mini_grids(self) -> List[str]:
        """
        Returns the site names or IDs of all mini-grid locations.

        Returns:
            A list of mini-grid site IDs.
        """
        if self._existing_minigrids_gdf.empty:
            print("Warning: No mini-grid data loaded.")
            return []
        if "Location" not in self._minigrids_gdf.columns:
            print(
                "Warning: 'Location' column not found in mini-grids data. Returning index."
            )
            return self._minigrids_gdf.index.astype(str).tolist()
        return self._existing_minigrids_gdf["Location"].tolist()

    def get_layer_geometry(
        self, layer_name: str, region: base.BaseGeometry
    ) -> Optional[Polygon]:
        """
        Returns the Shapely geometry for the union of features of a given layer.


        Uses the layer_map to access the appropriate GeoDataFrame based on layer_name.
            layer_map = {
            'buildings': self._buildings_gdf,
            'minigrids': self._minigrids_gdf,
            'tiles': self._joined_tiles_gdf, # Use the joined gdf for tile queries
            'roads': self._roads_gdf,
            'villages': self._villages_gdf,
            'parishes': self._parishes_gdf,
            'subcounties': self._subcounties_gdf,
            'existing_grid': self._existing_grid_gdf,
            'grid_extension': self._grid_extension_gdf,
            'candidate_minigrids': self._candidate_minigrids_gdf,
            'existing_minigrids': self._existing_minigrids_gdf
        }

        Args:
            layer_name: The name of the layer.
            region: The Shapely Polygon or Point defining the area of interest.

        Returns:
            The Shapely Polygon geometry, or None if the id is not found.
        """
        ### YOUR IMPLEMENTATION HERE ###
        gdf = self.get_gdf_info_within_region(region, layer_name)
        if gdf.empty:
            return None
        return gdf.geometry.union_all("unary")

    def nearest_mini_grids(self, pt: Point, k: int = 3) -> List[Tuple[str, float]]:
        """
        Finds the k closest mini-grid sites to a given point.

        Args:
            pt: The Shapely Point for the query location.
            k: The number of nearest mini-grids to return.

        Returns:
            A list of tuples (site_id, distance_meters). Returns an empty list
            if no mini-grids are available or an error occurs.
        """
        if self._existing_minigrids_gdf.empty:
            print("Warning: No mini-grid data loaded for nearest_mini_grids.")
            return []

        # Ensure minigrids GeoDataFrame is in a metric CRS for accurate distance calculation
        minigrids_metric = self._check_and_reproject_gdf(
            self._existing_minigrids_gdf.copy(), self.target_metric_crs
        )

        # Ensure the query point is also in the same metric CRS
        point_metric, _ = self._prepare_geometry_for_crs(pt, self.target_metric_crs)

        try:
            minigrids_metric.loc[:, "distance"] = minigrids_metric.geometry.distance(
                point_metric
            )
            nearest = minigrids_metric.nsmallest(k, "distance")

            if nearest.empty:
                return []

            if "site_id" not in nearest.columns:
                print(
                    "Warning: 'site_id' column not found for nearest mini-grids. Returning index."
                )
                return list(zip(nearest.index.astype(str), nearest["distance"]))

            return list(zip(nearest["site_id"], nearest["distance"]))
        except Exception as e:
            print(f"Error finding nearest mini-grids: {e}")
            return []

    def compute_distance_to_grid(self, geometry: base.BaseGeometry) -> float:
        """
        Computes the distance from a geometry to the nearest existing grid feature.

        Args:
            geometry: The Shapely geometry (Point, LineString, or Polygon) to measure distance from.

        Returns:
            The distance in meters to the nearest existing grid feature, or NaN if no grid features are available.
        """
        if self._existing_grid_gdf.empty:
            print("Warning: No existing grid data loaded for compute_distance_to_grid.")
            return float("nan")

        # Ensure existing grid GeoDataFrame is in a metric CRS for accurate distance calculation
        existing_grid_metric = self._check_and_reproject_gdf(
            self._existing_grid_gdf.copy(), self.target_metric_crs
        )

        # Ensure the input geometry is also in the same metric CRS
        geometry_metric, _ = self._prepare_geometry_for_crs(geometry, self.target_metric_crs)
        geometry = geometry_metric.geometry.iloc[0]

        try:
            distances = existing_grid_metric.geometry.distance(geometry)
            return int(distances.min()) if not distances.empty else float("nan")
        except Exception as e:
            print(f"Error computing distance to existing grid: {e}")
            return float("nan")

    # -----------------------------------------------------------------------------
    # 5) Region Analysis
    # -----------------------------------------------------------------------------
    def _analyze_settlements_in_region(self, region: Polygon) -> Dict[str, Any]:
        """Analyzes building data and settlement patterns within the region."""
        # Get buildings data
        buildings_gdf = self.get_gdf_info_within_region(region, "buildings")

        # Calculate total buildings
        total_buildings = len(buildings_gdf)

        # Calculate buildings by category
        building_categories = {}
        for category in buildings_gdf["category"].dropna().unique():
            count = len(buildings_gdf[buildings_gdf["category"] == category])
            building_categories[str(category)] = count

        # Get total residential buildings (those with no category)
        residential_count = len(buildings_gdf[buildings_gdf["category"].isna()])
        if residential_count > 0:
            building_categories["residential"] = residential_count

        # Get villages data
        villages_gdf = self.get_gdf_info_within_region(region, "villages")
        village_data = []

        # Process villages (limit to maximum 20 villages for LLM consumption)
        for _, village in villages_gdf.head(20).iterrows():
            village_info = {
                "name": village.get("addr_vname", "Unnamed"),
                "electrification_category": village.get("category", "Unknown"),
            }

            # Add rank information for candidate minigrids
            if village.get("category") == "Candidate minigrid" and not pd.isna(
                village.get("rank")
            ):
                village_info["priority_rank"] = int(village.get("rank"))

            village_data.append(village_info)

        return {
            "building_count": total_buildings,
            "building_categories": building_categories,
            "intersecting_village_count": len(villages_gdf),
            "intersecting_village_details": village_data,
            "has_truncated_villages": len(villages_gdf) > 20,
        }

    def _analyze_infrastructure_in_region(self, region: Polygon) -> Dict[str, Any]:
        """Analyzes infrastructure elements including roads, grid, and energy systems."""
        # Road analysis
        roads_gdf = self.get_gdf_info_within_region(region, "roads")
        road_types = {}

        for highway_type in roads_gdf["highway"].dropna().unique():
            count = len(roads_gdf[roads_gdf["highway"] == highway_type])
            road_types[str(highway_type)] = count

        # Grid infrastructure analysis
        grid_present = self.count_features_within_region(region, "existing_grid") > 0
        grid_extension = self.count_features_within_region(region, "grid_extension") > 0
        distance_to_grid = self.compute_distance_to_grid(region)

        # Candidate minigrids
        candidate_minigrids_gdf = self.get_gdf_info_within_region(
            region, "candidate_minigrids"
        )
        existing_minigrids_gdf = self.get_gdf_info_within_region(
            region, "existing_minigrids"
        )

        # Calculate total population to be served
        population_to_be_served = candidate_minigrids_gdf["Population"].sum()

        # Process capacity information
        capacity_data = {}
        for capacity in candidate_minigrids_gdf["capacity"].dropna().unique():
            count = len(
                candidate_minigrids_gdf[candidate_minigrids_gdf["capacity"] == capacity]
            )
            capacity_data[str(capacity)] = count

        return {
            "roads": {"total_road_segments": len(roads_gdf), "road_types": road_types},
            "electricity": {
                "existing_grid_present": grid_present,
                "distance_to_existing_grid": distance_to_grid,
                "grid_extension_proposed": grid_extension,
                "candidate_minigrids_count": len(candidate_minigrids_gdf),
                "existing_minigrids_count": len(existing_minigrids_gdf),
                "capacity_distribution": capacity_data,
                "population_to_be_served": int(population_to_be_served),
            },
        }

    def _analyze_administrative_divisions(self, region: Polygon) -> Dict[str, Any]:
        """Analyzes administrative boundaries within the region."""
        # Get parishes and subcounties
        parishes_gdf = self.get_gdf_info_within_region(region, "parishes")
        subcounties_gdf = self.get_gdf_info_within_region(region, "subcounties")

        # Process parish information
        parish_data = []
        for _, parish in parishes_gdf.iterrows():
            parish_data.append(
                {
                    "name": parish.get("addr_pname", "Unnamed"),
                    "electrification_category": parish.get("category", "Unknown"),
                }
            )

        # Process subcounty information
        subcounty_names = subcounties_gdf["addr_sname"].dropna().unique().tolist()

        return {
            "parishes": {"count": len(parishes_gdf), "details": parish_data},
            "subcounties": {"count": len(subcounties_gdf), "names": subcounty_names},
        }

    def _analyze_environmental_metrics(self, region: Polygon) -> Dict[str, Any]:
        """Analyzes environmental characteristics of the region."""
        # Get all weighted tile statistics
        env_stats = self.weighted_tile_stats_all(region)

        # Format and round numeric values for readability
        for key, value in env_stats.items():
            if isinstance(value, (int, float)):
                env_stats[key] = round(value, 4)

        # Add NDVI classification
        ndvi_value = env_stats.get("ndvi_mean", float("nan"))
        if not np.isnan(ndvi_value):
            if ndvi_value > 0.5:
                env_stats["vegetation_density"] = "Dense vegetation"
            elif ndvi_value > 0.2:
                env_stats["vegetation_density"] = "Moderate vegetation"
            elif ndvi_value > 0:
                env_stats["vegetation_density"] = "Sparse vegetation"
            else:
                env_stats["vegetation_density"] = "Very limited vegetation"
        # add total biomass for region
        env_stats["total_biomass_Mg"] = self.region_total_biomass(region)

        return env_stats

    def analyze_region(self, region: Polygon) -> Dict[str, Any]:
        """
        Performs comprehensive analysis of all geospatial layers within the specified region.

        Returns structured data about settlements, infrastructure, administrative boundaries,
        and environmental characteristics within the region of interest.

        Args:
            region: The Shapely Polygon defining the area of interest.

        Returns:
            A structured dictionary with comprehensive analysis results.
        """
        return {
            "settlements": self._analyze_settlements_in_region(region),
            "infrastructure": self._analyze_infrastructure_in_region(region),
            "administrative": self._analyze_administrative_divisions(region),
            "environment": self._analyze_environmental_metrics(region),
            # "region_summary": self._generate_region_summary(region)
        }

    
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
    def buffer_geometry(
        self, geom: base.BaseGeometry, radius_m: float
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
        if hasattr(geom, "crs") and geom.crs is not None:
            original_crs = geom.crs
        elif hasattr(geom, "index") and isinstance(
            geom.index, gpd.GeoSeries
        ):  # Check if it's a GeoSeries
            original_crs = geom.index.crs
        # Add other checks for how the geometry is represented and if it has CRS info

        geom_to_buffer = geom

        # If in geographic CRS (like WGS84), reproject to a suitable metric CRS (like UTM)
        # Need to know the appropriate UTM zone for the AOI (Lamwo, Uganda is around 36N)
        geographic_crs_codes = [4326]  # WGS84

        if original_crs and original_crs.to_epsg() in geographic_crs_codes:
            print(
                f"Reprojecting geometry from {original_crs} to {self.target_metric_crs} for buffering."
            )
            # Create a temporary GeoSeries to reproject the Shapely geometry
            temp_gs = gpd.GeoSeries([geom], crs=original_crs).to_crs(
                self.target_metric_crs
            )
            geom_to_buffer = temp_gs.iloc[0]
            buffered_geom = geom_to_buffer.buffer(radius_m)
            # Optional: Reproject back to the original CRS if needed
            # buffered_geom = gpd.GeoSeries([buffered_geom], crs=self.target_metric_crs).to_crs(original_crs).iloc[0]
            return buffered_geom
        else:
            # Assume the geometry is already in a suitable metric CRS
            # Or if no CRS info, perform buffer directly (less accurate)
            print(
                "Warning: Input geometry for buffering has no or non-geographic CRS. Buffering directly (accuracy depends on input CRS)."
            )
            return geom_to_buffer.buffer(radius_m)

    # -----------------------------------------------------------------------------
    # 8) Visualization Primitive (for verification/output)
    # -----------------------------------------------------------------------------
    def visualize_layers(
        self,
        center_point: Optional[Point] = None,
        zoom_start: int = 12,
        show_buildings: bool = False,
        show_minigrids: bool = True,
        show_tiles: bool = False,
        show_tile_stats: bool = False,  # Option to style tiles based on stats
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
        map_center = [0, 0]  # Default center

        if center_point is None:
            if not self._plain_tiles_gdf.empty:
                # Ensure the centroid calculation handles CRS
                try:
                    # Reproject to a suitable geographic CRS for Folium
                    tiles_for_centroid = self._check_and_reproject_gdf(
                        self._plain_tiles_gdf.copy(), self.target_geographic_crs
                    )
                    calculated_center = tiles_for_centroid.geometry.centroid.iloc[0]
                    map_center = [calculated_center.y, calculated_center.x]
                except Exception as e:
                    print(
                        f"Warning: Could not calculate map center from plain tiles: {e}. Using default center."
                    )
            else:
                print(
                    "Warning: Plain tiles data is empty. Cannot calculate map center. Using default center."
                )
                map_center = [0, 0]  # Use a global default if no data

        else:
            # Ensure the input center point is in a geographic CRS for Folium
            try:
                center_point_geographic, _ = self._prepare_geometry_for_crs(
                    center_point, self.target_geographic_crs
                )
                map_center = [center_point_geographic.y, center_point_geographic.x]
            except Exception as e:
                print(
                    f"Warning: Could not use provided center point due to CRS issues: {e}. Using default center."
                )
                map_center = [0, 0]

        # Create a base Folium map
        m = folium.Map(location=map_center, zoom_start=zoom_start)

        # Add layers based on parameters
        if show_tiles and not self._plain_tiles_gdf.empty:
            try:
                # Reproject to geographic CRS for Folium
                tiles_for_vis = self._check_and_reproject_gdf(
                    self._plain_tiles_gdf.copy(), self.target_geographic_crs
                )
                folium.GeoJson(
                    tiles_for_vis.to_json(),
                    name="Plain Tiles",
                    style_function=lambda feature: {
                        "fillColor": "none",
                        "color": "gray",
                        "weight": 1,
                    },
                ).add_to(m)
            except Exception as e:
                print(f"Error adding plain tiles to map: {e}")

        if show_minigrids and not self._candidate_minigrids_gdf.empty:
            try:
                # Reproject to geographic CRS for Folium
                minigrids_for_vis = self._check_and_reproject_gdf(
                    self._candidate_minigrids_gdf.copy(), self.target_geographic_crs
                )
                folium.GeoJson(
                    minigrids_for_vis.to_json(),
                    name="Mini Grids",
                    marker=folium.CircleMarker(
                        radius=5, fill=True, fill_color="red", color="red"
                    ),
                ).add_to(m)
            except Exception as e:
                print(f"Error adding mini grids to map: {e}")

        if show_buildings and not self._buildings_gdf.empty:
            # Note: Adding a large number of complex polygons can make the map slow.
            # Consider adding a subset or focusing on buildings within a smaller area if needed.
            try:
                # Reproject to geographic CRS for Folium
                buildings_for_vis = self._check_and_reproject_gdf(
                    self._buildings_gdf.copy(), self.target_geographic_crs
                )
                # Limit the number of buildings for performance if necessary
                # buildings_for_vis = buildings_for_vis.head(1000)
                folium.GeoJson(
                    buildings_for_vis.to_json(),
                    name="Buildings",
                    style_function=lambda feature: {
                        "fillColor": "blue",
                        "color": "blue",
                        "weight": 1,
                        "fillOpacity": 0.2,
                    },
                ).add_to(m)
            except Exception as e:
                print(f"Error adding buildings to map: {e}")

        if (
            show_tile_stats
            and not self._joined_tiles_gdf.empty
            and "ndvi_mean" in self._joined_tiles_gdf.columns
        ):
            try:
                # Reproject to geographic CRS for Folium
                tiles_stats_for_vis = self._check_and_reproject_gdf(
                    self._joined_tiles_gdf.copy(), self.target_geographic_crs
                )
                folium.GeoJson(
                    tiles_stats_for_vis.to_json(),
                    name="Tile Stats (NDVI)",
                    style_function=lambda feature: {
                        "fillColor": (
                            "green"
                            if feature["properties"].get("ndvi_mean", 0) > 0.4
                            else "orange"
                        ),
                        "color": (
                            "green"
                            if feature["properties"].get("ndvi_mean", 0) > 0.4
                            else "orange"
                        ),
                        "weight": 1,
                        "fillOpacity": 0.5,
                    },
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=["ndvi_mean"], aliases=["NDVI Mean:"]
                    ),
                ).add_to(m)
            except Exception as e:
                print(f"Error adding tile stats to map: {e}")
        elif show_tile_stats:
            print(
                "Warning: Cannot show tile stats. Joined tiles data is empty or missing 'ndvi_mean' column."
            )

        # Add layer control to switch layers on/off
        folium.LayerControl().add_to(m)

        # Display the map (this works automatically in a Colab cell)
        # display(m) # Uncomment this line if you need to explicitly display

        return m  # Return the map object
    def _get_layer_gdf(self, layer_name: str) -> gpd.GeoDataFrame:
        if layer_name not in self._layer_map:
            raise ValueError(
                f"Unknown layer name '{layer_name}'. Available layers: {list(self._layer_map.keys())}"
            )
        return self._layer_map[layer_name]

    def get_layer_bounds(self, layer_name: str) -> List[float]:
        gdf = self._get_layer_gdf(layer_name)
        if gdf.empty:
            return [float("nan")] * 4
        return gdf.total_bounds.tolist()

    def get_layer_geojson(
        self,
        layer_name: str,
        *,
        limit: Optional[int] = None,
        sample: Optional[int] = None,
        target_crs: str = "EPSG:4326",
    ) -> Dict[str, Any]:
        gdf = self._get_layer_gdf(layer_name)
        if gdf.empty:
            return {"type": "FeatureCollection", "features": []}

        subset = gdf
        if sample is not None and sample > 0:
            subset = subset.sample(min(sample, len(subset)))
        elif limit is not None and limit > 0:
            subset = subset.head(limit)

        if subset.crs is not None:
            crs_str = subset.crs.to_string() if hasattr(subset.crs, "to_string") else str(subset.crs)
            if crs_str != target_crs:
                subset = subset.to_crs(target_crs)
        else:
            subset = subset.set_crs(target_crs, allow_override=True)

        return json.loads(subset.to_json())

    def get_layer_count(self, layer_name: str) -> int:
        gdf = self._get_layer_gdf(layer_name)
        return int(len(gdf))
