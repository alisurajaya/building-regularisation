import geopandas as gpd
from geopandas import GeoDataFrame
from multiprocessing_class import MultiProcessing
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import json
from shapely.ops import unary_union
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait


class Regularisation:
    def __init__(self, data, file_path, output_geojson) -> None:
        self.file_path: str = file_path
        self.output_geojson:str = output_geojson
        self.data: GeoDataFrame = data
        self.mbr_gdf:GeoDataFrame = GeoDataFrame()
        self.grid_polygons_gdf: GeoDataFrame = GeoDataFrame()
        # self.multiprocessing = multiprocessing#: MultiProcessing = MultiProcessing()
        self.merged_gdf: GeoDataFrame = GeoDataFrame()
        self.result_gdf: GeoDataFrame = GeoDataFrame()

        # Configure logging
        logging.basicConfig(level=logging.DEBUG,  # Set the minimum level for the logger
                            format='%(asctime)s - %(levelname)s - %(message)s')  # Define the format for log messages

        # Create a logger
        self.logger = logging.getLogger(__name__)

    
    def create_mbr(self):
        self.logger.info("Create minimum bounding rectangle")
        # Create minimum bounding rectangles for each polygon
        mbr_geometries = []

        for fid, geometry in enumerate(self.data['geometry']):
            # if geometry.type == 'Polygon':
            if geometry.geom_type == 'Polygon':
                mbr = gpd.GeoDataFrame(geometry=[geometry.minimum_rotated_rectangle])
                mbr['id'] = fid  # Add FID attribute
                
                mbr_geometries.append(mbr)

        # Concatenate the list of GeoDataFrames into a single GeoDataFrame
        self.mbr_gdf = gpd.GeoDataFrame(pd.concat(mbr_geometries, ignore_index=True))
    

    
    def __calculate_angles(self, rectangle_coords):
        angles = []

        for i in range(4):
            x1, y1 = rectangle_coords[i]
            x2, y2 = rectangle_coords[(i + 1) % 4]

            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)

            angles.append(angle_deg)

        return angles


    def generate_grid(self, grid_size):
        self.logger.info("Generate Grid")

        # Set the grid size
        grid_size = grid_size

        # Generate Grids
        grid_data_list = {'geometry': [], 'id': []}
        for index, row in self.mbr_gdf.iterrows():
            if row['geometry'].geom_type == 'Polygon':
                polygon = row['geometry']

                # Read FID
                fid = row['id']
                # print(fid)

                # Read coordinates
                coordinates_list = list(polygon.exterior.coords)

                # Calculate rotation angle
                rotation_angle = self.__calculate_angles(coordinates_list)
                if rotation_angle[0] >= 50 or rotation_angle[0] <= -50:
                    rotation_angle = rotation_angle[1]
                else:
                    rotation_angle = rotation_angle[0]

                # Define cols based on grid size and bounding box
                xmin, ymin, xmax, ymax = polygon.bounds
                cols = int(np.ceil((xmax - xmin) / grid_size))
                rows = int(np.ceil((ymax - ymin) / grid_size))

                for i in range(cols):
                    for j in range(rows):
                        left = xmin + i * grid_size
                        right = xmin + (i + 1) * grid_size
                        bottom = ymin + j * grid_size
                        top = ymin + (j + 1) * grid_size

                        grid_polygon = Polygon([(left, bottom), (right, bottom), (right, top), (left, top)])
                        if grid_polygon.intersects(polygon):
                            intersection = grid_polygon.intersection(polygon)
                            if not intersection.is_empty:
                                grid_data_list['geometry'].append(intersection)
                                grid_data_list['id'].append(fid)

        # Create a GeoDataFrame from the grid data
        self.grid_polygons_gdf = gpd.GeoDataFrame(grid_data_list, columns=['geometry', 'id'])

        self.grid_polygons_gdf.crs = self.data.crs


    def calculate_overlap_area_per_feature(self, crs, gdf1, gdf2, spatial_index):
        results = []
        for index, feature1 in gdf1.iterrows():
            possible_matches_index = list(spatial_index.intersection(feature1['geometry'].bounds))
            possible_matches = gdf2.iloc[possible_matches_index]
            candidates = possible_matches[possible_matches.intersects(feature1['geometry'])]
            overlap_area = candidates.geometry.to_crs(crs).area.sum()

            results.append((index, overlap_area))

        return results
    
    def chunkify(self, chunk_size, gdf2, spatial_index):
        for index in range(0, len(self.grid_polygons_gdf), chunk_size):
            yield (self.grid_polygons_gdf.crs, self.grid_polygons_gdf[index:index+chunk_size], gdf2, spatial_index)

    def filter(self, area_threshold, unique_ids, id_field):
        merged_geometries = []
        for id_val in unique_ids:
            selected_grids = self.result_gdf[(self.result_gdf[id_field] == id_val) & (self.result_gdf['overlap_area'] > area_threshold)]
            if not selected_grids.empty:
                merged_geometry = unary_union(selected_grids['geometry'])
                merged_geometries.append(merged_geometry)

        return merged_geometries

    def filter_chunkify(self, chunk_size, unique_ids, area_threshold, id_field):
        for index in range(0, len(unique_ids), chunk_size):
            yield (area_threshold, unique_ids[index:index+chunk_size], id_field)
    

    def grid_selection(self, threshold_percentage, id_field):
        self.logger.info("Grid Selection")

        multiprocessing.freeze_support()
        # Define the number of processes to use
        num_processes = multiprocessing.cpu_count()

        """
            # Calculate overlap area per feature
        """
        self.logger.warning("Calculate overlap area per feature")
        gdf2 = self.data 

        # Create a spatial index for gdf2
        spatial_index = gdf2.sindex

        # Reproject gdf2 to match the CRS of gdf1
        gdf2 = gdf2.to_crs(self.grid_polygons_gdf.crs)

        # Create a new GeoDataFrame to store the results
        result_gdf = self.grid_polygons_gdf.copy()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(self.calculate_overlap_area_per_feature, self.chunkify(500, gdf2, spatial_index))

        # Create a DataFrame from results for efficient lookup
        results_df = pd.DataFrame([(index, overlap_area) for sublist in results for index, overlap_area in sublist],
                                columns=['index', 'overlap_area'])
        # Set index as 'index' column for efficient lookup
        results_df.set_index('index', inplace=True)
        # Update 'overlap_area' column in result_gdf using loc accessor
        result_gdf.loc[results_df.index, 'overlap_area'] = results_df['overlap_area']

        self.result_gdf = result_gdf

        # result_gdf = self.__calculate_overlap_area_per_feature()
        """
            # Filter grids with overlap area more than 25% of the grid area
        """
        self.logger.warning("filter") 
        
        # Set the threshold percentage for selecting grids (e.g., 25%)
        threshold_percentage = threshold_percentage

        # Specify the ID field name for grids
        id_field = id_field

        # Precompute area threshold
        area_threshold = result_gdf['geometry'].area * threshold_percentage / 100
        unique_ids = result_gdf[id_field].unique()

        with multiprocessing.Pool(processes=num_processes) as pool:
            chunks = self.filter_chunkify(500, unique_ids, area_threshold, id_field)
            merged_geometries = pool.starmap(self.filter, chunks)
        results = [geometry for sub_geometry in merged_geometries for geometry in sub_geometry]

        self.merged_gdf = gpd.GeoDataFrame(geometry=results, crs=result_gdf.crs)


    def save_to_file(self):
        self.logger.info("Save the result as a file")
        # Save the merged geometries to a new GeoJSON file
        self.merged_gdf.to_file(self.output_geojson, driver='GeoJSON')

        # Define CRS as input
        crs = json.load(open(self.file_path))['crs']
        with open(self.output_geojson, "r") as jsonFile:
            data = json.load(jsonFile)

        data["crs"] = crs

        with open(self.output_geojson, "w") as jsonFile:
            json.dump(data, jsonFile)