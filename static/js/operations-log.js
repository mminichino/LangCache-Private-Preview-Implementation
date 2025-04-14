document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const logQuery = document.getElementById('log-query');
    const logModel = document.getElementById('log-model');
    const logTimestamp = document.getElementById('log-timestamp');
    const logResult = document.getElementById('log-result');
    const logResultContainer = document.getElementById('log-result-container');
    const logTotalTime = document.getElementById('log-total-time');
    const logSteps = document.getElementById('log-steps');

    // Function to fetch operations log
    function fetchOperationsLog() {
        fetch('/operations-log')
            .then(response => response.json())
            .then(data => {
                updateOperationsLog(data);
            })
            .catch(error => {
                console.error('Error fetching operations log:', error);
            });
    }

    // Function to update the operations log display
    function updateOperationsLog(data) {
        // Update query info
        logQuery.textContent = data.query || 'No query executed yet';
        logModel.textContent = data.embedding_model || '-';
        logTimestamp.textContent = data.timestamp || '-';

        // Update result summary
        if (data.result && Object.keys(data.result).length > 0) {
            if (data.result.cache_hit) {
                logResult.textContent = 'CACHE HIT';
                logResult.className = 'value hit';
                if (data.result.similarity) {
                    logResult.textContent += ` (Similarity: ${parseFloat(data.result.similarity).toFixed(2)})`;
                }
            } else {
                logResult.textContent = 'CACHE MISS';
                logResult.className = 'value miss';
            }
            
            if (data.result.total_time) {
                logTotalTime.textContent = typeof data.result.total_time === 'string' 
                    ? data.result.total_time 
                    : `${data.result.total_time.toFixed(3)}s`;
            } else {
                logTotalTime.textContent = '-';
            }
        } else {
            logResult.textContent = '-';
            logResult.className = 'value';
            logTotalTime.textContent = '-';
        }

        // Update steps
        if (data.steps && data.steps.length > 0) {
            // Clear existing steps
            logSteps.innerHTML = '';
            
            // Add each step
            data.steps.forEach(step => {
                const stepElement = document.createElement('div');
                stepElement.className = `log-step ${getStepClass(step.step)}`;
                
                const stepHeader = document.createElement('div');
                stepHeader.className = 'step-header';
                stepHeader.innerHTML = `
                    <span class="step-time">[${step.timestamp || ''}]</span>
                    <span class="step-name">${step.step}</span>
                `;
                
                const stepDetails = document.createElement('div');
                stepDetails.className = 'step-details';
                
                // Add details
                if (step.details && Object.keys(step.details).length > 0) {
                    const detailsList = document.createElement('ul');
                    
                    for (const [key, value] of Object.entries(step.details)) {
                        const detailItem = document.createElement('li');
                        detailItem.innerHTML = `<span class="detail-key">${formatKey(key)}:</span> <span class="detail-value">${value}</span>`;
                        detailsList.appendChild(detailItem);
                    }
                    
                    stepDetails.appendChild(detailsList);
                }
                
                stepElement.appendChild(stepHeader);
                stepElement.appendChild(stepDetails);
                logSteps.appendChild(stepElement);
            });
        } else {
            logSteps.innerHTML = '<p class="no-steps">No operations logged yet. Try running a query in the Demo tab.</p>';
        }
    }

    // Helper function to get CSS class for step type
    function getStepClass(stepType) {
        switch(stepType) {
            case 'QUERY PROCESSING':
                return 'step-query';
            case 'EMBEDDING GENERATION':
                return 'step-embedding';
            case 'REDIS VECTOR SEARCH':
                return 'step-redis';
            case 'CACHE RESULT':
                return 'step-result';
            case 'RESPONSE RETRIEVAL':
                return 'step-response';
            case 'LLM GENERATION':
                return 'step-llm';
            case 'CACHE STORAGE':
                return 'step-storage';
            case 'RESPONSE':
                return 'step-response';
            case 'ERROR':
                return 'step-error';
            default:
                return '';
        }
    }

    // Helper function to format detail keys
    function formatKey(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    // Fetch operations log when the tab is shown
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            if (this.getAttribute('data-tab') === 'operations-log') {
                fetchOperationsLog();
            }
        });
    });

    // Also fetch when a query is submitted (with a slight delay to ensure log is updated)
    document.getElementById('submit-query').addEventListener('click', function() {
        // Wait a bit for the query to complete and log to be updated
        setTimeout(fetchOperationsLog, 500);
    });

    // Initial fetch if the operations log tab is active
    if (document.querySelector('.tab-button[data-tab="operations-log"]').classList.contains('active')) {
        fetchOperationsLog();
    }
});
