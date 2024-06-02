import pandas as pd
import json
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from shapely.ops import unary_union
import time
import multiprocessing
import logging

"""
    1. Create Minimum Bounding Rectangle (MBR) ###
"""
def create_mbr(data):
    # Create minimum bounding rectangles for each polygon
    mbr_geometries = []

    for fid, geometry in enumerate(data['geometry']):
        if geometry.geom_type == 'Polygon':
            mbr = gpd.GeoDataFrame(geometry=[geometry.minimum_rotated_rectangle])
            mbr['id'] = fid  # Add FID attribute
            mbr_geometries.append(mbr)

    # Concatenate the list of GeoDataFrames into a single GeoDataFrame
    mbr_gdf = gpd.GeoDataFrame(pd.concat(mbr_geometries, ignore_index=True))

    return mbr_gdf

"""
    2. Generate Grids ###
"""
def calculate_angles(rectangle_coords):
    angles = []

    for i in range(4):
        x1, y1 = rectangle_coords[i]
        x2, y2 = rectangle_coords[(i + 1) % 4]

        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)

        angles.append(angle_deg)

    return angles


def create_rotated_grid(polygon, grid_size, rotation_angle, fid):
    xmin, ymin, xmax, ymax = polygon.bounds
    rows = int(np.ceil((ymax - ymin) / grid_size))
    cols = int(np.ceil((xmax - xmin) / grid_size))

    centroid = polygon.centroid

    grid_data = {'geometry': [], 'id': []}
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
                    grid_data['geometry'].append(intersection)
                    grid_data['id'].append(fid)

    return grid_data


def generate_grid(gdf, data_crs, grid_size):
    # Generate Grids
    grid_data_list = {'geometry': [], 'id': []}
    for index, row in gdf.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            polygon = row['geometry']

            # Read FID
            fid = row['id']
            # print(fid)

            # Read coordinates
            coordinates_list = list(polygon.exterior.coords)

            # Calculate rotation angle
            rotation_angle = calculate_angles(coordinates_list)
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
    grid_polygons_gdf = gpd.GeoDataFrame(grid_data_list, columns=['geometry', 'id'])

    grid_polygons_gdf.crs = data_crs

    return grid_polygons_gdf



"""
    3. Grid Selection ###
"""
def check_the_data(args):
    index, feature1, gdf2, spatial_index, gdf1_crs = args
    possible_matches_index = list(spatial_index.intersection(feature1['geometry'].bounds))
    possible_matches = gdf2.iloc[possible_matches_index]
    candidates = possible_matches[possible_matches.intersects(feature1['geometry'])]
    overlap_area = candidates.geometry.to_crs(gdf1_crs).area.sum()
    return index, overlap_area

def calculate_overlap_area_per_feature(gdf1, gdf2):
    # Create a spatial index for gdf2
    spatial_index = gdf2.sindex

    # Reproject gdf2 to match the CRS of gdf1
    gdf1_crs = gdf1.crs
    gdf2 = gdf2.to_crs(gdf1_crs)

    # Create a new GeoDataFrame to store the results
    result_gdf = gdf1.copy()

    """
        multiprocessing part
    """
    # Define the number of processes to use
    num_processes = multiprocessing.cpu_count() 
    # Create argument tuples for each row
    args = [(index, feature1, gdf2, spatial_index, gdf1_crs) for index, feature1 in gdf1.iterrows()]

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            # Apply the processing function to each row in parallel
            results = pool.map(check_the_data, args)
        except KeyboardInterrupt:
            if pool:
                pool.terminate()
    # Update the result GeoDataFrame
    for index, overlap_area in results:
        result_gdf.at[index, 'overlap_area'] = overlap_area

    return result_gdf

def filter(grid_gdf, threshold_percentage, id_field):
    merged_geometries = []
    # Filter and merge grids with overlap area more than threshold_percentage of the grid area
    for id_val in grid_gdf[id_field].unique():
        selected_grids = grid_gdf[(grid_gdf[id_field] == id_val) & (grid_gdf['overlap_area'] > grid_gdf['geometry'].area * threshold_percentage / 100)]
        if not selected_grids.empty:
            merged_geometry = unary_union(selected_grids['geometry'])
            merged_geometries.append(merged_geometry)

    # Create a new GeoDataFrame for the merged geometries
    merged_gdf = gpd.GeoDataFrame(geometry=merged_geometries, crs=grid_gdf.crs)

    return merged_gdf


def grid_selection(gdf1, gdf2, file_path):
    # Reproject gdf2 to match the CRS of gdf1
    gdf2 = gdf2.to_crs(crs=gdf1.crs)

    result_gdf = calculate_overlap_area_per_feature(gdf1, gdf2)

    # Set the threshold percentage for selecting grids (e.g., 25%)
    threshold_percentage = 50

    # Specify the ID field name for grids
    id_field = 'id'

    # Filter grids with overlap area more than 25% of the grid area
    return filter(result_gdf, threshold_percentage, id_field)

def save_to_file(merged_gdf, file_path, output_geojson):
    features = []
    count = 0
    for index, row in merged_gdf.iterrows():
        geom = row.geometry
        geom_json = geom.__geo_interface__
        
        # feature structure
        geojson = {
            'type':'Feature',
            'properties':{
                'id': count,
                'kelas':'bangunan'
            },
            'geometry': {
                'type': geom_json["type"],
                'coordinates': geom_json["coordinates"]
            }
        }
        features.append(geojson)
        count += 1

    feature_coll = {
        'type':'FeatureCollection',
        'name': output_geojson,
        'crs':{
            'type':'name',
            'properties':{
                'name':f'urn:ogc:def:crs:EPSG::{merged_gdf.crs.to_epsg()}'
            }
        },
        'features': features
    }

    with open(output_geojson, 'w') as f:
        f.write(json.dumps(feature_coll, indent=2))



def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG,  # Set the minimum level for the logger
                        format='%(asctime)s - %(levelname)s - %(message)s')  # Define the format for log messages

    # Create a logger
    logger = logging.getLogger(__name__)

    # Record the start time
    start_time = time.time()

    # Path to your input shapefile
    file_path = 'files/renggang.geojson'

    # Specify the output GeoJSON file for selected grids
    output_geojson = 'files/renggang-mp.geojson'

    # Read the shapefile using geopandas
    data = gpd.read_file(file_path)

    logger.info("Create minimum bounding rectangle")
    mbr_gdf = create_mbr(data)

    logger.info("Generate Grid")
    grid_polygons_gdf = generate_grid(gdf=mbr_gdf, data_crs=data.crs, grid_size=0.5)

    logger.info("Grid Selection")
    merged_gdf = grid_selection(gdf1=grid_polygons_gdf, gdf2=data, file_path=file_path)

    logger.info("Save the result as a file")
    save_to_file(merged_gdf, file_path, output_geojson)

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_minutes = elapsed_time/60

    print(f"Processing time: {elapsed_time} seconds or {time_minutes} minutes")

if __name__ == "__main__":
    main()