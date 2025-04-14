# Redis Langcache Demo

A demonstration application showcasing Redis Langcache for semantic caching of LLM responses.

## Overview

This application demonstrates the power of Redis Langcache for semantic caching of Large Language Model (LLM) responses. It provides a web interface that allows users to:

1. Submit queries directly to LLMs (without caching)
2. Submit queries through Redis Langcache (with semantic caching)
3. Compare performance metrics between cached and non-cached responses
4. Analyze query matches and cache operations
5. View detailed operation logs for each query

## Features

- **Semantic Caching**: Uses vector similarity to find semantically similar queries in the cache
- **Multiple Embedding Models**: Supports multiple embedding models for comparison:
  - Redis Langcache-Embed (default)
  - OpenAI Embeddings
  - Ollama BGE-3
- **Performance Metrics**: Tracks and displays latency metrics for:
  - Cache operations (embedding generation + Redis search)
  - LLM response generation
  - Cache hit rates
- **Query Analysis**: Allows users to analyze which queries match in the cache and their similarity scores
- **Operations Log**: Provides detailed logs of each step in the query processing pipeline
- **Settings Management**: Customize similarity thresholds and other parameters

## Architecture

The application consists of several components:

1. **Web Application**: A Flask application that provides the user interface and handles query processing
2. **Redis Langcache Services**: Three instances of Redis Langcache, one for each embedding model
3. **Custom Embeddings Service**: A FastAPI service that provides embeddings using the redis/langcache-embed-v1 model
4. **Redis Database**: Stores the vector embeddings and cached responses

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Redis Stack
- Hugging Face API token (for custom embeddings)
- Google AI API key (for Gemini LLM)

## Installation

### Using Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/redis-langcache-demo.git
   cd redis-langcache-demo
   ```

2. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   GITHUB_TOKEN=your_github_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   A template file `.env.example` is provided for reference.

3. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the application at http://localhost:5001

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/redis-langcache-demo.git
   cd redis-langcache-demo
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start Redis:
   ```bash
   docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
   ```

4. Start the custom embeddings service:
   ```bash
   cd custom-embeddings
   python main.py
   ```

5. Start the Redis Langcache services:
   ```bash
   # Start Ollama BGE service
   docker run -d --name langcache-ollama -p 8080:8080 artifactory.dev.redislabs.com:443/cloud-docker-dev-local/ai-services/langcache:0.0.7

   # Start Redis Langcache service
   docker run -d --name langcache-redis -p 8081:8080 artifactory.dev.redislabs.com:443/cloud-docker-dev-local/ai-services/langcache:0.0.7

   # Start OpenAI service
   docker run -d --name langcache-openai -p 8082:8080 artifactory.dev.redislabs.com:443/cloud-docker-dev-local/ai-services/langcache:0.0.7
   ```

6. Start the main application:
   ```bash
   python app.py
   ```

7. Access the application at http://localhost:5001

## Usage

### Direct LLM Queries

1. Select the "Direct LLM" panel
2. Enter your query
3. Choose an LLM model (e.g., Gemini 1.5 Flash)
4. Click "Submit"

### Redis Langcache Queries

1. Select the "Redis Langcache" panel
2. Enter your query
3. Choose an embedding model (Redis Langcache-Embed is the default)
4. Click "Submit"

### Viewing Metrics

1. Click on the "Latency" tab to view performance metrics
2. See cache hit rates and latency metrics for each embedding model
3. Compare performance between different models

### Query Analysis

1. Click on the "Query Analysis" tab
2. View the history of queries and their cache matches
3. Filter by embedding model
4. See similarity scores for each match

### Operations Log

1. Click on the "Operations Log" tab
2. Select a query from the dropdown
3. View detailed logs of each step in the query processing pipeline

## How It Works

1. **Direct LLM Queries**:
   - User submits a query
   - Application sends the query directly to the LLM
   - LLM generates a response
   - Response is displayed to the user

2. **Redis Langcache Queries**:
   - User submits a query
   - Application generates an embedding for the query
   - Application searches Redis for similar queries
   - If a similar query is found (cache hit):
     - Cached response is returned
   - If no similar query is found (cache miss):
     - Application sends the query to the LLM
     - LLM generates a response
     - Response is stored in the cache
     - Response is displayed to the user

## Cache Operations Breakdown

When using Redis Langcache, the cache operations consist of two main components:

1. **Embedding Generation**: Converting the query text into a vector embedding
   - This is typically the most time-consuming part (>99% of the total cache operation time)
   - Different embedding models have different performance characteristics

2. **Redis Search**: Searching for similar vectors in Redis
   - This is extremely fast, typically less than 1ms
   - Uses vector similarity search to find the most similar cached queries

The application provides a detailed breakdown of these operations in the Latency tab.

## Performance Considerations

- **Embedding Generation**: The most time-consuming part of the caching process
- **Redis Search**: Very fast, typically less than 1ms
- **Cache Hit Rate**: Depends on the similarity threshold (default: 0.85) and the variety of queries
- **LLM Latency**: Varies by model and query complexity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Redis Labs for Redis Langcache
- Hugging Face for the embedding models
- Google for the Gemini LLM API
