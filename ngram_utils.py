import re
from collections import defaultdict, Counter

# Cache for storing n-grams of cached queries
cached_query_ngrams = {}
ngram_frequencies = Counter()

def preprocess_text(text):
    """Preprocess text by lowercasing and removing special characters"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def generate_ngrams(text, n=3):
    """Generate n-grams from text"""
    text = preprocess_text(text)
    words = text.split()
    
    # Generate n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    # If text is too short for n-grams, use smaller n-grams
    if not ngrams and len(words) > 0:
        if len(words) == 1:
            ngrams = words
        else:
            for i in range(len(words) - 1):
                ngram = ' '.join(words[i:i+2])
                ngrams.append(ngram)
    
    return ngrams

def update_ngram_cache(query, query_id):
    """Update the n-gram cache with a new query"""
    # Generate n-grams for the query
    ngrams = generate_ngrams(query)
    
    # Store n-grams for this query
    cached_query_ngrams[query_id] = ngrams
    
    # Update n-gram frequencies
    for ngram in ngrams:
        ngram_frequencies[ngram] += 1

def calculate_ngram_similarity(query, cached_queries):
    """Calculate n-gram similarity between query and cached queries"""
    query_ngrams = generate_ngrams(query)
    
    best_match_score = 0
    best_match_id = None
    
    for query_id, cached_query in cached_queries.items():
        # Get or generate n-grams for cached query
        if query_id in cached_query_ngrams:
            cached_ngrams = cached_query_ngrams[query_id]
        else:
            cached_ngrams = generate_ngrams(cached_query)
            cached_query_ngrams[query_id] = cached_ngrams
        
        # Calculate Jaccard similarity with weighting
        overlap = 0
        for ngram in query_ngrams:
            if ngram in cached_ngrams:
                # Weight by inverse frequency (rare n-grams count more)
                weight = 1.0 / (ngram_frequencies[ngram] + 1)
                overlap += weight
        
        # Calculate total unique n-grams
        total_ngrams = len(set(query_ngrams).union(cached_ngrams))
        
        # Calculate similarity score
        if total_ngrams > 0:
            similarity = overlap / total_ngrams
            
            # Update best match if this is better
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = query_id
    
    return best_match_score, best_match_id

def estimate_cache_hit_probability(query, cached_queries):
    """Estimate the probability of a cache hit for a query"""
    if not cached_queries:
        return 0.0
    
    # Calculate n-gram similarity
    similarity, _ = calculate_ngram_similarity(query, cached_queries)
    
    # Convert similarity to probability using sigmoid-like function
    # This can be calibrated based on historical data
    probability = min(1.0, similarity * 1.5)
    
    return probability
