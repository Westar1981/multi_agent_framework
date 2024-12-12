#!/bin/bash

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install necessary packages
echo "Installing necessary packages..."
sudo apt install -y python3 python3-pip git python3-venv

# Set up a virtual environment
echo "Setting up a virtual environment..."
python3 -m venv env
source env/bin/activate

# Initialize a Git repository
echo "Initializing Git repository..."
git init

# Install Python packages
echo "Installing Python packages..."
pip install tensorflow flask

# Create configuration files
echo "Creating configuration files..."
touch .env
echo "FLASK_APP=app.py" >> .env
echo "FLASK_ENV=development" >> .env

echo "Development environment setup complete!"
