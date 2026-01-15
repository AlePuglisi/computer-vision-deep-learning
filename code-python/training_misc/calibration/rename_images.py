import os
import argparse

def rename_images(folder_path):
    try:
        # Get all files in the folder
        image_files = sorted(os.listdir(folder_path))  # Sorting ensures consistent order

        for i, filename in enumerate(image_files):
            new_name = f"image{i}.jpg"  # Format with leading zeros
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_name}'")

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    cwd = os.getcwd()
    dataset_path = os.path.join(cwd,"images")
    split_type = "test"
    split_path = os.path.join(dataset_path, split_type)
    class_name = "person"
    full_dataset_path = os.path.join(split_path, class_name)
    image_folder_path = './gopro_hero3/calib_images/'
    rename_images(image_folder_path)