document.addEventListener('DOMContentLoaded', function() {
    // Add event listener for latency tab
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            if (this.getAttribute('data-tab') === 'latency') {
                // Load latency data when tab is shown
                fetchLatencyData();
            }
        });
    });

    // Initial load if latency tab is active
    if (document.querySelector('.tab-button[data-tab="latency"].active')) {
        fetchLatencyData();
    }

    // Function to fetch latency data from the server
    function fetchLatencyData() {
        fetch('/latency-data')
            .then(response => response.json())
            .then(data => {
                updateCharts(data);
                updateMetrics(data);
            })
            .catch(error => {
                console.error('Error fetching latency data:', error);
            });
    }

    // No charts to update
    function updateCharts(data) {
        // Charts removed
    }

    // Function to update metrics summary
    function updateMetrics(data) {
        // Update metrics for each embedding model
        updateModelMetrics('ollama-bge', data['ollama-bge']);
        updateModelMetrics('redis-langcache', data['redis-langcache']);
    }

    // Helper function to update metrics for a specific model
    function updateModelMetrics(modelId, modelData) {
        if (!modelData) return;

        // Update cache operation latency (total)
        document.getElementById(`${modelId}-cache-latency`).textContent =
            modelData.current_cache_latency ? modelData.current_cache_latency.toFixed(2) + 's' : '-';

        // Update embedding latency
        document.getElementById(`${modelId}-embedding-latency`).textContent =
            modelData.current_embedding_latency ? modelData.current_embedding_latency.toFixed(2) + 's' : '-';

        // Update Redis search latency
        document.getElementById(`${modelId}-redis-latency`).textContent =
            modelData.current_redis_latency ? modelData.current_redis_latency.toFixed(3) + 's' : '-';

        // Update cache hit rate
        document.getElementById(`${modelId}-cache-hit-rate`).textContent =
            modelData.cache_hit_rate ? (modelData.cache_hit_rate * 100).toFixed(1) + '%' : '-';
    }

    // No auto-refresh - data will be updated when a new query is submitted
});
