import os

def print_folder_structure(start_path, indent=""):
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, "").count(os.sep)
        indent_space = "│   " * level
        print(f"{indent_space}├── 📁 {os.path.basename(root)}/")
        for f in files:
            print(f"{indent_space}│   ├── 📄 {f}")
        dirs[:] = sorted(dirs)  # Sort directories for consistent order
        files[:] = sorted(files)  # Sort files as well
        # Don't recurse here since os.walk handles it

# Example usage:
print_folder_structure(".")

