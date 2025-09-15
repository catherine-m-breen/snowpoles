import os
import tarfile
import sys

def extract_tar_gz_files(source_dir):
    """
    Walks through the specified directory and extracts all .tar.gz files
    """
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        return False
    
    # Counter for successful extractions
    extracted_count = 0
    failed_files = []
    
    print(f"Starting extraction of .tar.gz files in: {source_dir}")
    
    # Walk through the directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.tar.gz'):
                tar_path = os.path.join(root, file)
                extract_dir = os.path.join(root, os.path.splitext(os.path.splitext(file)[0])[0])
                
                # Create extraction directory if it doesn't exist
                if not os.path.exists(extract_dir):
                    os.makedirs(extract_dir)
                
                try:
                    print(f"Extracting: {tar_path} to {extract_dir}")
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(path=extract_dir)
                    extracted_count += 1
                except Exception as e:
                    print(f"Failed to extract {tar_path}: {str(e)}")
                    failed_files.append(tar_path)
    
    # Summary
    print("\nExtraction Summary:")
    print(f"Total .tar.gz files successfully extracted: {extracted_count}")
    
    if failed_files:
        print(f"Failed to extract {len(failed_files)} files:")
        for file in failed_files:
            print(f"  - {file}")
    
    return True

if __name__ == "__main__":
    source_directory = '/Volumes/My Book/snex17'
    extract_tar_gz_files(source_directory)