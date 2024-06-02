from regularisation import Regularisation
import time
import geopandas as gpd
import cProfile

def main():
    # Path to your input shapefile
    file_path = 'files/bogor1x1.geojson'
    output_geojson = 'files/bogor1x1-reg.geojson'

    # Read the shapefile using geopandas
    data = gpd.read_file(file_path)

    regularisation = Regularisation(data=data, file_path=file_path, output_geojson=output_geojson)

    # Record the start time
    start_time = time.time()

    regularisation.create_mbr()
    regularisation.generate_grid(grid_size=0.5)
    # cProfile.runctx("regularisation.grid_selection()", globals(), locals())
    regularisation.grid_selection(threshold_percentage=50, id_field="id")
    regularisation.save_to_file()

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_minutes = elapsed_time/60

    print(f"Processing time: {elapsed_time} seconds or {time_minutes} minutes")

if __name__ == "__main__":
    main()