from flask import Flask, render_template, request, jsonify
import time
import random
import json
import os
import requests
import datetime
import statistics
from collections import defaultdict
from dotenv import load_dotenv
from google import genai
import logging

# Load environment variables from .env file (if available)
# In GCP, these will come from environment variables
load_dotenv(override=False)

app = Flask(__name__)

# Initialize the Gemini API client
# For GCP, this will be set by Secret Manager
API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
genai_client = genai.Client(api_key=API_KEY)

# Get Redis URL from environment variable (set by Cloud Run)
REDIS_URL = os.environ.get('REDIS_URL')
if not REDIS_URL:
    print("WARNING: REDIS_URL environment variable not set!")

# Redis Langcache API configuration
LANGCACHE_INDEX_NAME = 'gemini_cache'

# Define URLs for different embedding models (adjusted for Cloud Run)
CUSTOM_EMBEDDINGS_URL = os.environ.get('CUSTOM_EMBEDDINGS_URL', 'http://embeddings:8080')

LANGCACHE_URLS = {
    'ollama-bge': os.environ.get('OLLAMA_EMBEDDINGS_URL', 'http://langcache-ollama:8080'),
    'redis-langcache': os.environ.get('REDIS_LANGCACHE_URL', 'http://langcache-redis:8080')
}

# Define cache IDs for different embedding models
cache_ids = {}
# Default embedding model
DEFAULT_EMBEDDING_MODEL = 'redis-langcache'

# Latency tracking
latency_data = {
    'ollama-bge': {
        'llm': defaultdict(list),
        'cache': defaultdict(list),
        'embedding': defaultdict(list),
        'redis': defaultdict(list),
        'cache_hits': 0,
        'cache_misses': 0
    },
    'redis-langcache': {
        'llm': defaultdict(list),
        'cache': defaultdict(list),
        'embedding': defaultdict(list),
        'redis': defaultdict(list),
        'cache_hits': 0,
        'cache_misses': 0
    }
}

# Operations log for the most recent query
operations_log = {
    'query': '',
    'embedding_model': '',
    'cache_id': '',
    'timestamp': '',
    'steps': [],
    'result': {}
}

# Query match tracking
query_matches = []
# Format: [
#   {
#     'timestamp': '2025-04-04 12:00',
#     'query': 'user query',
#     'matched_query': 'cache query',
#     'model': 'embedding model name',
#     'similarity': 0.95,
#     'embedding_time': 0.5,  # seconds
#     'cache_id': 'cache entry id'
#   },
#   ...
# ]

# Cache queries for n-gram analysis
cached_queries = {}
# Format: {
#   'cache_id': 'query text',
#   ...
# }

# File logger for cache operations
import json as _json
cache_log_path = os.path.join(os.path.dirname(__file__), '..', 'cache_operations.log')
def log_cache_event(event_type, details):
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'event': event_type,
        'details': details
    }
    with open(cache_log_path, 'a') as f:
        f.write(_json.dumps(log_entry) + '\n')

def get_current_timestamp():
    """Get current timestamp in format 'YYYY-MM-DD HH:MM' rounded to nearest 15 min"""
    now = datetime.datetime.now()
    # Round to nearest 15 minutes
    minute = 15 * (now.minute // 15)
    rounded_time = now.replace(minute=minute, second=0, microsecond=0)
    return rounded_time.strftime('%Y-%m-%d %H:%M')

# Redis Langcache API functions
def create_cache():
    """Create caches for each embedding model"""
    global cache_ids

    for model_name, base_url in LANGCACHE_URLS.items():
        url = f"{base_url}/v1/admin/caches"
        payload = {
            "indexName": f"{LANGCACHE_INDEX_NAME}_{model_name}",
            "redisUrls": [REDIS_URL],  # Use the REDIS_URL from environment
            "overwriteIfExists": True,
            "allowExistingData": True,
            "defaultSimilarityThreshold": 0.85
        }

        try:
            print(f"Creating cache for {model_name} at {base_url}...")
            response = requests.post(url, json=payload)
            if response.status_code == 200 or response.status_code == 201:
                data = response.json()
                cache_id = data.get('cacheId')
                cache_ids[model_name] = cache_id
                print(f"Cache for {model_name} created with ID: {cache_id}")
                log_cache_event('CACHE_CREATE', {'model': model_name, 'cache_id': cache_id, 'status': 'success'})
            else:
                print(f"Error creating cache for {model_name}: {response.status_code} - {response.text}")
                log_cache_event('CACHE_CREATE', {'model': model_name, 'status': 'error', 'error': response.text})
        except Exception as e:
            print(f"Error creating cache for {model_name}: {e}")
            log_cache_event('CACHE_CREATE', {'model': model_name, 'status': 'exception', 'error': str(e)})

    # Return True if at least one cache was created successfully
    return len(cache_ids) > 0

def search_cache(query, embedding_model="ollama-bge"):
    """Search for a similar query in the cache using the specified embedding model"""
    global operations_log

    # Reset operations log for new query
    operations_log = {
        'query': query,
        'embedding_model': embedding_model,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'steps': [],
        'result': {}
    }

    # Log the query processing step
    operations_log['steps'].append({
        'step': 'QUERY PROCESSING',
        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
        'details': {
            'user_query': query,
            'embedding_model': embedding_model
        }
    })

    # Get the cache ID for the selected embedding model
    cache_id = cache_ids.get(embedding_model)
    if not cache_id:
        print(f"No cache_id available for {embedding_model}, skipping cache search")
        operations_log['steps'].append({
            'step': 'ERROR',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'message': f"No cache_id available for {embedding_model}"
            }
        })
        log_cache_event('CACHE_SEARCH', {'query': query, 'model': embedding_model, 'result': 'error', 'reason': 'no_cache_id'})
        return None

    # Store cache ID in operations log
    operations_log['cache_id'] = cache_id

    # Get the base URL for the selected embedding model
    base_url = LANGCACHE_URLS.get(embedding_model)
    if not base_url:
        print(f"No base URL available for {embedding_model}, skipping cache search")
        operations_log['steps'].append({
            'step': 'ERROR',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'message': f"No base URL available for {embedding_model}"
            }
        })
        log_cache_event('CACHE_SEARCH', {'query': query, 'model': embedding_model, 'result': 'error', 'reason': 'no_base_url'})
        return None

    url = f"{base_url}/v1/caches/{cache_id}/search"
    payload = {
        "prompt": query,
        "similarityThreshold": 0.85  # Updated threshold as requested
    }

    # Track embedding generation time separately
    embedding_start_time = time.time()

    try:
        print(f"Searching cache for query: {query} with model: {embedding_model}")
        log_cache_event('CACHE_SEARCH', {'query': query, 'model': embedding_model, 'cache_id': cache_id, 'status': 'started'})

        # Log embedding generation step
        operations_log['steps'].append({
            'step': 'EMBEDDING GENERATION',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'model': embedding_model,
                'service_url': base_url,
                'cache_id': cache_id
            }
        })

        # First part: Generate embeddings and send request
        response = requests.post(url, json=payload)
        embedding_time = time.time() - embedding_start_time

        # Second part: Process response (Redis search time)
        redis_search_start_time = time.time()

        # Track embedding generation time for the specific embedding model
        timestamp = get_current_timestamp()
        latency_data[embedding_model]['embedding'][timestamp].append(embedding_time)

        # Update embedding step with timing information
        operations_log['steps'][-1]['details']['embedding_time'] = f"{embedding_time:.3f}s"

        # Log Redis search step
        operations_log['steps'].append({
            'step': 'REDIS VECTOR SEARCH',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'similarity_threshold': payload['similarityThreshold'],
                'index_name': f"{LANGCACHE_INDEX_NAME}_{embedding_model}"
            }
        })

        print(f"Cache search response status: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Cache search response data: {data}")

                # According to the API spec, the response is an array of entries
                if data and len(data) > 0:
                    # Return the most similar entry (first one)
                    entry = data[0]
                    similarity = entry['similarity']
                    print(f"Cache hit found with similarity: {similarity}")

                    # Calculate Redis search time
                    redis_search_time = time.time() - redis_search_start_time
                    print(f"Redis search processing time: {redis_search_time:.6f}s (excluding embedding generation)")

                    # Track Redis search time for the specific embedding model
                    latency_data[embedding_model]['redis'][timestamp].append(redis_search_time)

                    # Update Redis search step with timing
                    operations_log['steps'][-1]['details']['redis_search_time'] = f"{redis_search_time:.3f}s"

                    # Log similarity results
                    operations_log['steps'].append({
                        'step': 'CACHE RESULT',
                        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                        'details': {
                            'top_match_similarity': f"{similarity:.2f}",
                            'original_query': entry['prompt'],
                            'result': 'CACHE HIT',
                            'above_threshold': True
                        }
                    })

                    # Track cache hit for the specific embedding model
                    latency_data[embedding_model]['cache_hits'] += 1

                    # Log response retrieval
                    operations_log['steps'].append({
                        'step': 'RESPONSE RETRIEVAL',
                        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                        'details': {
                            'source': 'Cache',
                            'total_time': f"{embedding_time + redis_search_time:.3f}s"
                        }
                    })

                    # Update result summary
                    operations_log['result'] = {
                        'cache_hit': True,
                        'similarity': similarity,
                        'total_time': embedding_time + redis_search_time
                    }

                    log_cache_event('CACHE_HIT', {'query': query, 'model': embedding_model, 'cache_id': cache_id, 'matches': data})

                    return {
                        'response': entry['response'],
                        'similarity': similarity,
                        'entryId': entry['id'],
                        'matched_query': entry['prompt'],  # Add the matched query
                        'embedding_time': embedding_time,
                        'redis_search_time': redis_search_time
                    }
                else:
                    print("No similar entries found in cache")

                    # Calculate Redis search time
                    redis_search_time = time.time() - redis_search_start_time

                    # Update Redis search step with timing
                    operations_log['steps'][-1]['details']['redis_search_time'] = f"{redis_search_time:.3f}s"

                    # Log cache miss
                    operations_log['steps'].append({
                        'step': 'CACHE RESULT',
                        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                        'details': {
                            'result': 'CACHE MISS',
                            'message': 'No matches found in cache'
                        }
                    })

                    # Update result summary
                    operations_log['result'] = {
                        'cache_hit': False,
                        'total_time': embedding_time + redis_search_time
                    }

                    log_cache_event('CACHE_MISS', {'query': query, 'model': embedding_model, 'cache_id': cache_id})
            except ValueError as json_error:
                print(f"Error parsing JSON response: {json_error}")
                print(f"Response content: {response.text[:100]}...")
                # Calculate Redis search time for error case
                redis_search_time = time.time() - redis_search_start_time
                return {
                    'embedding_time': embedding_time,
                    'redis_search_time': redis_search_time
                }
        else:
            print(f"Cache search failed with status {response.status_code}: {response.text}")
        # Calculate Redis search time for cache miss
        redis_search_time = time.time() - redis_search_start_time
        print(f"Redis search processing time (cache miss): {redis_search_time:.6f}s (excluding embedding generation)")

        # Track Redis search time for the specific embedding model
        latency_data[embedding_model]['redis'][timestamp].append(redis_search_time)

        return {
            'embedding_time': embedding_time,
            'redis_search_time': redis_search_time
        }
    except Exception as e:
        print(f"Error searching cache: {e}")
        return None

def add_to_cache(query, response, embedding_model="ollama-bge"):
    """Add a new entry to the cache using the specified embedding model"""
    global operations_log

    # Log LLM generation step if this is called after a cache miss
    if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
        # This is the same query, so we're adding to cache after a miss
        operations_log['steps'].append({
            'step': 'LLM GENERATION',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'model': 'gemini-1.5-flash',  # This is hardcoded for now
                'message': 'Generated response from LLM'
            }
        })

    # Get the cache ID for the selected embedding model
    cache_id = cache_ids.get(embedding_model)
    if not cache_id:
        print(f"No cache_id available for {embedding_model}, skipping cache addition")
        if operations_log.get('query') == query:
            operations_log['steps'].append({
                'step': 'ERROR',
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'details': {
                    'message': f"No cache_id available for {embedding_model}"
                }
            })
        return None

    # Get the base URL for the selected embedding model
    base_url = LANGCACHE_URLS.get(embedding_model)
    if not base_url:
        print(f"No base URL available for {embedding_model}, skipping cache addition")
        if operations_log.get('query') == query:
            operations_log['steps'].append({
                'step': 'ERROR',
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'details': {
                    'message': f"No base URL available for {embedding_model}"
                }
            })
        return None

    url = f"{base_url}/v1/caches/{cache_id}/entries"
    payload = {
        "prompt": query,
        "response": response
    }

    try:
        print(f"Adding entry to cache for query: {query[:50]}... with model: {embedding_model}")

        # Log cache storage step if this is the same query
        if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
            operations_log['steps'].append({
                'step': 'CACHE STORAGE',
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'details': {
                    'message': 'Storing new query and response in cache',
                    'model': embedding_model
                }
            })

        # Make the API call
        cache_storage_start = time.time()
        resp = requests.post(url, json=payload)
        cache_storage_time = time.time() - cache_storage_start
        print(f"Cache add response status: {resp.status_code}")

        if resp.status_code == 200 or resp.status_code == 201:
            data = resp.json()
            entry_id = data.get('entryId')
            print(f"Successfully added entry to cache with ID: {entry_id}")

            # If we have an entry ID, store the query for n-gram analysis
            if entry_id:
                cached_queries[entry_id] = query
                print(f"Added query to n-gram cache with ID: {entry_id}")

                # Update cache storage step if this is the same query
                if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
                    operations_log['steps'][-1]['details']['entry_id'] = entry_id
                    operations_log['steps'][-1]['details']['storage_time'] = f"{cache_storage_time:.3f}s"

                    # Add response step
                    operations_log['steps'].append({
                        'step': 'RESPONSE',
                        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                        'details': {
                            'source': 'LLM',
                            'message': 'Response generated by LLM and stored in cache'
                        }
                    })

            log_cache_event('CACHE_ADD', {'query': query, 'model': embedding_model, 'cache_id': cache_id, 'status': 'started'})
            if resp.status_code == 200 or resp.status_code == 201:
                log_cache_event('CACHE_ADD', {'query': query, 'model': embedding_model, 'cache_id': cache_id, 'status': 'success'})
                return entry_id
            else:
                log_cache_event('CACHE_ADD', {'query': query, 'model': embedding_model, 'cache_id': cache_id, 'status': 'error', 'error': resp.text})
                return None
        else:
            print(f"Failed to add to cache: {resp.status_code} - {resp.text}")

            # Log error if this is the same query
            if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
                operations_log['steps'].append({
                    'step': 'ERROR',
                    'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                    'details': {
                        'message': f"Failed to add to cache: {resp.status_code}"
                    }
                })

            return None
    except Exception as e:
        print(f"Error adding to cache: {e}")

        # Log error if this is the same query
        if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
            operations_log['steps'].append({
                'step': 'ERROR',
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'details': {
                    'message': f"Error adding to cache: {e}"
                }
            })

        return None

# Initialize cache on startup
try:
    create_cache()
except Exception as e:
    print(f"Warning: Could not initialize cache: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query', '')
    use_cache = data.get('use_cache', False)
    llm_model = data.get('llm_model', 'gemini-1.5-flash')
    embedding_model = data.get('embedding_model', 'ollama-bge')

    print(f"Processing query: '{query}', use_cache: {use_cache}, llm_model: {llm_model}, embedding_model: {embedding_model}")

    if use_cache:
        # This is the semantic cache panel
        # First, check if we have a similar query in the Redis semantic cache
        cache_start_time = time.time()
        cached_result = search_cache(query, embedding_model)
        cache_time = time.time() - cache_start_time

        # Get embedding and Redis search times if available
        embedding_time = 0
        redis_search_time = 0
        if cached_result and 'embedding_time' in cached_result:
            embedding_time = cached_result['embedding_time']
        if cached_result and 'redis_search_time' in cached_result:
            redis_search_time = cached_result['redis_search_time']

        # The cache search time is the total time for the cache operation
        # This includes embedding generation + searching in Redis + returning results

        # For debugging
        print(f"Total cache time: {cache_time:.4f}s, Embedding time: {embedding_time:.4f}s, Redis search time: {redis_search_time:.6f}s")
        print(f"Percentage breakdown - Embedding: {(embedding_time/cache_time)*100:.2f}%, Redis search: {(redis_search_time/cache_time)*100:.2f}%, Other: {((cache_time-embedding_time-redis_search_time)/cache_time)*100:.2f}%")

        # Track total cache operation latency for the specific embedding model
        timestamp = get_current_timestamp()
        latency_data[embedding_model]['cache'][timestamp].append(cache_time)

        if cached_result and 'response' in cached_result:
            # We found a similar query in the cache
            similarity = cached_result.get('similarity', 'N/A')
            print(f"Cache hit! Returning cached response with similarity {similarity}")
            # Track cache hit for the specific embedding model
            latency_data[embedding_model]['cache_hits'] += 1

            # Track query match for analysis
            if 'entryId' in cached_result:
                # Get the matched query from the cache entry ID
                matched_query = cached_result.get('matched_query', 'Unknown')
                # Add to query matches list
                query_matches.append({
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'query': query,
                    'matched_query': matched_query,
                    'model': embedding_model,
                    'similarity': similarity,
                    'embedding_time': cached_result.get('embedding_time', 0),
                    'cache_id': cached_result.get('entryId', 'Unknown')
                })

            return jsonify({
                'response': cached_result['response'],
                'source': 'cache',
                'time_taken': cache_time,
                'similarity': similarity
            })

        # No cache hit, need to call the LLM and store the result
        print("Cache miss. Calling LLM and storing result...")
        # Track cache miss for the specific embedding model
        latency_data[embedding_model]['cache_misses'] += 1

        llm_start_time = time.time()
        response = generate_gemini_response(query, llm_model)
        llm_time = time.time() - llm_start_time

        # Calculate total time (cache search + LLM)
        total_time = cache_time + llm_time

        # Track LLM latency for the specific embedding model
        timestamp = get_current_timestamp()
        latency_data[embedding_model]['llm'][timestamp].append(llm_time)

        # Update operations log with LLM timing
        if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
            # Find the LLM generation step if it exists
            llm_step = next((step for step in operations_log['steps'] if step['step'] == 'LLM GENERATION'), None)

            if llm_step:
                # Update existing step
                llm_step['details']['llm_response_time'] = f"{llm_time:.3f}s"
                llm_step['details']['model'] = llm_model
            else:
                # Add new step
                operations_log['steps'].append({
                    'step': 'LLM GENERATION',
                    'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                    'details': {
                        'model': llm_model,
                        'llm_response_time': f"{llm_time:.3f}s"
                    }
                })

            # Update result summary
            operations_log['result']['total_time'] = total_time

        # Store this response in the cache for future use
        try:
            add_to_cache(query, response, embedding_model)
        except Exception as e:
            print(f"Error adding response to cache: {e}")

        return jsonify({
            'response': response,
            'source': 'llm',
            'time_taken': total_time  # Return the total time
        })

def generate_gemini_response(query, model_name="gemini-1.5-flash"):
    """Generate a response using Google's Gemini LLM"""
    try:
        # Use Google Gemini
        print(f"Calling Gemini API with query: {query}, model: {model_name}")
        # Call the actual Gemini API
        response = genai_client.models.generate_content(
            model=model_name,
            contents=[query]
        )
        print(f"Gemini API response received from model: {model_name}")
        return response.text
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        # Fallback to a generic response if there's an error
        return f"I encountered an issue processing your query about '{query}'. Please try again later."

@app.route('/latency-data')
def get_latency_data():
    """Return latency data for all embedding models"""
    # Get data for the current interval
    current_timestamp = get_current_timestamp()

    # Prepare result data structure
    result = {}

    # Process each embedding model
    for model_name in ['ollama-bge', 'redis-langcache']:
        model_data = latency_data[model_name]

        # Calculate current interval averages
        current_llm_latency = None
        current_cache_latency = None
        current_embedding_latency = None
        current_redis_latency = None
        cache_hit_rate = 0

        # LLM latency
        if current_timestamp in model_data['llm'] and model_data['llm'][current_timestamp]:
            current_llm_latency = statistics.mean(model_data['llm'][current_timestamp])

        # Cache operation latency
        if current_timestamp in model_data['cache'] and model_data['cache'][current_timestamp]:
            current_cache_latency = statistics.mean(model_data['cache'][current_timestamp])

        # Embedding generation latency
        if current_timestamp in model_data['embedding'] and model_data['embedding'][current_timestamp]:
            current_embedding_latency = statistics.mean(model_data['embedding'][current_timestamp])

        # Redis search latency
        if current_timestamp in model_data['redis'] and model_data['redis'][current_timestamp]:
            current_redis_latency = statistics.mean(model_data['redis'][current_timestamp])

        # Cache hit rate
        total_cache_requests = model_data['cache_hits'] + model_data['cache_misses']
        if total_cache_requests > 0:
            cache_hit_rate = model_data['cache_hits'] / total_cache_requests

        # Store metrics for this model
        result[model_name] = {
            'current_llm_latency': round(current_llm_latency, 3) if current_llm_latency is not None else None,
            'current_cache_latency': round(current_cache_latency, 3) if current_cache_latency is not None else None,
            'current_embedding_latency': round(current_embedding_latency, 3) if current_embedding_latency is not None else None,
            'current_redis_latency': round(current_redis_latency, 6) if current_redis_latency is not None else None,
            'cache_hit_rate': round(cache_hit_rate, 2) if cache_hit_rate > 0 else 0
        }

    return jsonify(result)

@app.route('/query-analysis')
def get_query_analysis():
    """Return query match data for analysis"""
    # Get the embedding model filter from query parameters
    model_filter = request.args.get('model', None)

    # Filter query matches by model if specified
    if model_filter and model_filter in ['ollama-bge', 'redis-langcache']:
        filtered_matches = [match for match in query_matches if match['model'] == model_filter]
    else:
        filtered_matches = query_matches

    # Sort by timestamp (newest first)
    sorted_matches = sorted(filtered_matches, key=lambda x: x['timestamp'], reverse=True)

    # Return the data
    return jsonify({
        'matches': sorted_matches,
        'total': len(sorted_matches),
        'models': ['ollama-bge', 'redis-langcache']
    })

@app.route('/operations-log')
def get_operations_log():
    """Return the operations log for the most recent query"""
    return jsonify(operations_log)


if __name__ == '__main__':
    # Create caches for each embedding model
    create_cache()
    # In GCP Cloud Run, the PORT environment variable is provided by the platform
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)