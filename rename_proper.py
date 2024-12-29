import os
from PIL import Image

def process_directory(dirpath):
    """Processes a single directory to convert JPG to PNG, remove JPGs, and rename PNGs sequentially."""
    # First, process all files in dirpath
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        if os.path.isfile(filepath):
            file_lower = filename.lower()
            if file_lower.endswith('.jpg') or file_lower.endswith('.jpeg'):
                # Convert to .png
                image = Image.open(filepath)
                base_name = os.path.splitext(filename)[0]  # Remove extension
                new_filename = base_name + '.png'
                new_filepath = os.path.join(dirpath, new_filename)
                image.save(new_filepath, 'PNG')
                os.remove(filepath)  # Remove original .jpg file
    
    # Get list of .png files
    png_files = [f for f in os.listdir(dirpath) if f.lower().endswith('.png')]
    png_files.sort()  # Sort alphabetically

    # Rename files to temporary names to avoid conflicts
    temp_files = []
    for filename in png_files:
        filepath = os.path.join(dirpath, filename)
        temp_filename = filename + '_temp'
        temp_filepath = os.path.join(dirpath, temp_filename)
        os.rename(filepath, temp_filepath)
        temp_files.append(temp_filename)

    # Rename temp files to sequential numbers
    for index, temp_filename in enumerate(temp_files, start=1):
        temp_filepath = os.path.join(dirpath, temp_filename)
        new_filename = f'{index}.png'
        new_filepath = os.path.join(dirpath, new_filename)
        os.rename(temp_filepath, new_filepath)

def process_all_directories_incrementally(root_dir):
    """Processes directories incrementally based on the pattern '<number>-2-real'."""
    increment = 0
    while True:
        folder_name = f'{increment}-2-real'
        dirpath = os.path.join(root_dir, folder_name)
        if os.path.exists(dirpath):
            print(f"Processing directory: {dirpath}")
            process_directory(dirpath)
            increment += 1
        else:
            print(f"No folder found for: {folder_name}. Stopping.")
            break

# Example usage
if __name__ == '__main__':
    root_directory = '/home/ubuntu/verb-workspace/WAVES/data/attacked/imagenet'  # Replace with your folder path
    process_all_directories_incrementally(root_directory)