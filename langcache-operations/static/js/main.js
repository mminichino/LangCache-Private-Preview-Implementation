document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');

            // Hide all tab contents
            tabContents.forEach(content => content.style.display = 'none');

            // Show the selected tab content
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId + '-content').style.display = 'block';
        });
    });

    const queryInput = document.getElementById('query-input');
    const submitButton = document.getElementById('submit-query');
    const cachedChat = document.getElementById('cached-chat');
    const standardChat = document.getElementById('standard-chat');
    const cachedTimeDisplay = document.getElementById('cached-time');
    const standardTimeDisplay = document.getElementById('standard-time');
    const llmModelSelect = document.getElementById('llm-model');
    const embeddingModelSelect = document.getElementById('embedding-model');

    // Clear welcome messages when first query is submitted
    let isFirstQuery = true;

    submitButton.addEventListener('click', handleSubmit);
    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSubmit();
        }
    });

    function handleSubmit() {
        const query = queryInput.value.trim();
        if (!query) return;

        // Always clear previous messages for each new query
        cachedChat.innerHTML = '';
        standardChat.innerHTML = '';
        if (isFirstQuery) {
            isFirstQuery = false;
        }

        // Add user message to both panels
        addMessage(cachedChat, query, 'user');
        addMessage(standardChat, query, 'user');

        // Add loading indicators
        const cachedLoadingMsg = addLoadingMessage(cachedChat);
        const standardLoadingMsg = addLoadingMessage(standardChat);

        // Reset time displays
        cachedTimeDisplay.textContent = '';
        standardTimeDisplay.textContent = '';

        // Start timers
        const cachedStartTime = performance.now();
        const standardStartTime = performance.now();

        // Make requests to both endpoints

        // 1. Cached version
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                use_cache: true,
                llm_model: llmModelSelect.value,
                embedding_model: embeddingModelSelect.value
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            cachedChat.removeChild(cachedLoadingMsg);

            // Use server-reported time for more accurate demonstration
            const timeTaken = data.time_taken.toFixed(2);

            // Display response time
            cachedTimeDisplay.textContent = `${timeTaken}s`;

            // Add response message
            addResponseMessage(cachedChat, data.response, data.source === 'cache', data.similarity);
        })
        .catch(error => {
            cachedChat.removeChild(cachedLoadingMsg);
            addErrorMessage(cachedChat, 'Error with semantic cache: ' + error.message);
            console.error('Error with semantic cache:', error);
        });

        // Clear input
        queryInput.value = '';
    }

    function addMessage(container, text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        messageDiv.textContent = text;
        container.appendChild(messageDiv);
        // Scroll to the top like ChatGPT does
        container.scrollTop = 0;
        return messageDiv;
    }

    function addResponseMessage(container, text, fromCache, similarity) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'response');

        // Create a wrapper for the content
        const contentWrapper = document.createElement('div');

        // Add source indicator at the top
        const sourceIndicator = document.createElement('div');
        sourceIndicator.classList.add('source-indicator');
        if (fromCache) {
            sourceIndicator.classList.add('cache');
            let cacheText = '✓ Retrieved from Redis semantic cache';
            if (similarity) {
                // Format similarity as percentage
                const similarityPercent = Math.round(similarity * 100);
                cacheText += ` (${similarityPercent}% match)`;
            }
            sourceIndicator.textContent = cacheText;
        } else {
            sourceIndicator.textContent = 'Generated by LLM';
        }

        // Add the response text
        const responseText = document.createElement('div');
        responseText.textContent = text;

        // Add elements to the message div in the correct order
        messageDiv.appendChild(sourceIndicator);
        messageDiv.appendChild(responseText);

        container.appendChild(messageDiv);
        // Scroll to the top like ChatGPT does
        container.scrollTop = 0;
        return messageDiv;
    }

    function addLoadingMessage(container) {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'loading', 'response');

        // Use the same simple loading animation for both panels
        const loadingDots = document.createElement('div');
        loadingDots.classList.add('loading-dots');

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            loadingDots.appendChild(dot);
        }

        loadingDiv.appendChild(loadingDots);

        container.appendChild(loadingDiv);
        // Scroll to the top like ChatGPT does
        container.scrollTop = 0;
        return loadingDiv;
    }

    function addErrorMessage(container, text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'response', 'error');
        messageDiv.textContent = text;
        container.appendChild(messageDiv);
        // Scroll to the top like ChatGPT does
        container.scrollTop = 0;
        return messageDiv;
    }

    // Function to update latency and query analysis data after a query completes
    function updateDataAfterQuery() {
        // Update latency data if the latency tab exists and the function is available
        if (typeof fetchLatencyData === 'function' && document.getElementById('latency-content')) {
            fetchLatencyData();
        }

        // Update query analysis data if the query analysis tab exists and the function is available
        if (typeof fetchQueryMatches === 'function' && document.getElementById('query-analysis-content')) {
            fetchQueryMatches();
        }
    }
});
