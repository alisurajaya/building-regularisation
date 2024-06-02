import geopandas as gpd
from geopandas import GeoDataFrame
from multiprocessing_class import MultiProcessing
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import json
from shapely.ops import unary_union, cascaded_union
import logging
import multiprocessing
import time
from shapely.geometry import MultiPolygon

    
def create_mbr(data):
    # Create minimum bounding rectangles for each polygon
    mbr_geometries = []

    for fid, geometry in enumerate(data['geometry']):
        # if geometry.type == 'Polygon':
        if geometry.geom_type == 'Polygon':
            mbr = gpd.GeoDataFrame(geometry=[geometry.minimum_rotated_rectangle])
            mbr['id'] = fid  # Add FID attribute
            
            mbr_geometries.append(mbr)

    # Concatenate the list of GeoDataFrames into a single GeoDataFrame
    return gpd.GeoDataFrame(pd.concat(mbr_geometries, ignore_index=True))


def calculate_angles(rectangle_coords):
    angles = []

    for i in range(4):
        x1, y1 = rectangle_coords[i]
        x2, y2 = rectangle_coords[(i + 1) % 4]

        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)

        angles.append(angle_deg)

    return angles


def generate_grid(mbr_gdf, data_crs, grid_size):
    # Generate Grids
    grid_data_list = {'geometry': [], 'id': []}
    for index, row in mbr_gdf.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            polygon = row['geometry']

            # Read FID
            fid = row['id']

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


def calculate_overlap_area_per_feature(crs, gdf1, gdf2, spatial_index):
    results = []
    for index, feature1 in gdf1.iterrows():
        possible_matches_index = list(spatial_index.intersection(feature1['geometry'].bounds))
        possible_matches = gdf2.iloc[possible_matches_index]
        candidates = possible_matches[possible_matches.intersects(feature1['geometry'])]
        overlap_area = candidates.geometry.to_crs(crs).area.sum()

        results.append((index, overlap_area))

    return results

def chunkify(grid_polygons_gdf, chunk_size, gdf2, spatial_index):
    for index in range(0, len(grid_polygons_gdf), chunk_size):
        yield (grid_polygons_gdf.crs, grid_polygons_gdf[index:index+chunk_size], gdf2, spatial_index)


def filter(result_gdf, area_threshold, unique_ids, id_field):
    merged_geometries = []
    for id_val in unique_ids:
        selected_grids = result_gdf[(result_gdf[id_field] == id_val) & (result_gdf['overlap_area'] > area_threshold)]
        if not selected_grids.empty:
            # option for merging geometries (unary_union(), obj.convex_hull, cascaed_union())
            merged_geometry = unary_union(selected_grids['geometry'])
            merged_geometries.append(merged_geometry)

    return merged_geometries

def filter_chunkify(result_gdf, chunk_size, unique_ids, area_threshold, id_field):
    for index in range(0, len(unique_ids), chunk_size):
        yield (result_gdf, area_threshold, unique_ids[index:index+chunk_size], id_field)


def grid_selection(grid_polygons_gdf, gdf2, threshold_percentage, id_field):
    multiprocessing.freeze_support()
    # Define the number of processes to use
    num_processes = multiprocessing.cpu_count()

    """
        # Calculate overlap area per feature
    """

    # Create a spatial index for gdf2
    spatial_index = gdf2.sindex

    # Reproject gdf2 to match the CRS of gdf1
    gdf2 = gdf2.to_crs(grid_polygons_gdf.crs)

    # Create a new GeoDataFrame to store the results
    result_gdf = grid_polygons_gdf.copy()
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunks = chunkify(grid_polygons_gdf, 500, gdf2, spatial_index)
        results = pool.starmap(calculate_overlap_area_per_feature, chunks)

    # Create a DataFrame from results for efficient lookup
    results_df = pd.DataFrame([(index, overlap_area) for sublist in results for index, overlap_area in sublist],
                            columns=['index', 'overlap_area'])
    # Set index as 'index' column for efficient lookup
    results_df.set_index('index', inplace=True)
    # Update 'overlap_area' column in result_gdf using loc accessor
    result_gdf.loc[results_df.index, 'overlap_area'] = results_df['overlap_area']

    """
        # Filter grids with overlap area more than 25% of the grid area
    """
    # Precompute area threshold
    area_threshold = result_gdf['geometry'].area * threshold_percentage / 100
    unique_ids = result_gdf[id_field].unique()

    with multiprocessing.Pool(processes=num_processes) as pool:
        chunks = filter_chunkify(result_gdf, 500, unique_ids, area_threshold, id_field)
        merged_geometries = pool.starmap(filter, chunks)
    results = [geometry for sub_geometry in merged_geometries for geometry in sub_geometry]

    return gpd.GeoDataFrame(geometry=results, crs=result_gdf.crs)


def save_to_file(merged_gdf, file_path, output_geojson):
    # Save the merged geometries to a new GeoJSON file
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

    # Path to your input shapefile
    file_path = 'files/renggang.geojson'
    output_geojson = 'files/renggang-reg.geojson'

    # Read the shapefile using geopandas
    data = gpd.read_file(file_path)

    # regularisation = Regularisation(data=data, file_path=file_path, output_geojson=output_geojson)

    # Record the start time
    start_time = time.time()

    logger.info("Create minimum bounding rectangle")
    mbr_gdf = create_mbr(data)
    logger.info("Generate Grid")
    grid_polygons_gdf = generate_grid(mbr_gdf, data_crs=data.crs, grid_size=0.5)
    logger.info("Grid Selection")
    merged_gdf = grid_selection(grid_polygons_gdf, gdf2=data, threshold_percentage=50, id_field="id")
    logger.info("Save the result as a file")
    save_to_file(merged_gdf, file_path, output_geojson)
    logger.info("Finished")

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_minutes = elapsed_time/60

    print(f"Processing time: {elapsed_time} seconds or {time_minutes} minutes")

if __name__ == "__main__":
    main()