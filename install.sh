#!/bin/bash

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Move vision_utilities to a temporary location using rsync
sudo rsync -a vision_utilities/ ../vision_utilities_temp || handle_error "Failed to move vision_utilities to temporary location"

# Ensure the temporary directory exists
if [ ! -d "../vision_utilities_temp" ]; then
    handle_error "Temporary directory ../vision_utilities_temp does not exist"
fi

# Remove the original vision_utilities directory
sudo rm -rf vision_utilities || handle_error "Failed to remove original vision_utilities directory"

# Move the temporary vision_utilities back to the original location using rsync
sudo rsync -a ../vision_utilities_temp/ ../vision_utilities || handle_error "Failed to move vision_utilities_temp back to vision_utilities"

# Remove the temporary directory
sudo rm -rf ../vision_utilities_temp || handle_error "Failed to remove temporary directory ../vision_utilities_temp"

echo "Operation completed successfully."