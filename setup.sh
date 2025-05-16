#!/bin/bash

# Setup script for LangCache Demo

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit the .env file and add your API keys."
else
    echo ".env file already exists."
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "Setup complete! You can now run the LangCache demo with:"
echo "docker-compose up -d langcache-redis embeddings llm-app"
echo ""
echo "Then visit http://localhost:5001 to access the demo application."
echo "The cache log is available at http://localhost:5001/log"
