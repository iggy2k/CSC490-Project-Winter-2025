import matplotlib.pyplot as plt
import geopandas as gpd
import json
from shapely.geometry import shape, GeometryCollection
from geojson import dump
import os
from tqdm import tqdm
from pathlib import Path


# https://stackoverflow.com/questions/56605238/how-to-use-tqdm-for-json-file-load-progress-bar
def hook(obj):
    value = obj.get("features")
    if not value:
        return obj
    pbar = tqdm(value)
    for _ in pbar:
        pbar.set_description(f"Loading")
    return obj


def simplify_geojson(path: str, tolerance: float):

    new_filename = f'{path}_compressed_{tolerance}.geojson'

    new_file = Path(new_filename)
    if new_file.is_file():
        print(f'{new_filename} alredy exists, skipping...')
        return None

    with open(path) as f:
        features = json.load(f, object_hook=hook)["features"]

    file_stats = os.stat(path)
    print(f'{path} Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')

    geom = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])
    polys = []
    pre = 0
    post = 0
    for p in geom.geoms:
        poly = p
        if poly.geom_type == 'Polygon':
            x = len(poly.exterior.coords)
        elif poly.geom_type == 'MultiPolygon':
            x = sum([len(poly_item.exterior.coords) for poly_item in poly.geoms])
        pre += x

        poly = poly.simplify(tolerance=tolerance)
        if poly.geom_type == 'Polygon':
            a = len(poly.exterior.coords)
        elif poly.geom_type == 'MultiPolygon':
            a = sum([len(poly_item.exterior.coords) for poly_item in poly.geoms])
        post += a

        polys.append(poly)

    print(f'Simplified {pre} polygons to {post}')
    p = gpd.GeoSeries(polys)
    p.plot()
    plt.show()

    with open(new_filename, 'w') as f:
        dump(GeometryCollection(polys), f)

    file_stats = os.stat(f'{path}_compressed_{tolerance}.geojson')
    print(f'{new_filename} Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')

    return new_filename