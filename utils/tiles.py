from functions import slice_raster

slice_raster(
    raster_in_path="/home/thomas/Documents/thomas/HighRes-GeoSpatial-Annotation/data/Nosy_boraha_nord_v2.tif",
    raster_out_dir="../sam/data/sud2/tiles1024",
    tile_size=1024, 
    skip_empty_tiles=True, # skip tiles that are empty (all values are the same, e.g. all zeros)
)

