from src.utils.file_utils import slice_raster

def tile_raster(raster_in_path, raster_out_dir, tile_size=1024, skip_empty_tiles=True):
    """
    Slices a raster into smaller tiles.

    Args:
        raster_in_path (str): Path to the input raster file.
        raster_out_dir (str): Directory to save the output tiles.
        tile_size (int): The size of the tiles.
        skip_empty_tiles (bool): Whether to skip empty tiles.
    """
    slice_raster(
        raster_in_path=raster_in_path,
        raster_out_dir=raster_out_dir,
        tile_size=tile_size, 
        skip_empty_tiles=skip_empty_tiles,
    )
