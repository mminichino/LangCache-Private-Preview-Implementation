import os
import datetime
import json
import threading
from collections import deque

class LogManager:
    """
    Manages a structured log file in Markdown format for tracking cache operations and queries.
    """
    def __init__(self, log_file_path="cache_log.md", max_queries=500, focus_model="redis-langcache"):
        """
        Initialize the log manager.

        Args:
            log_file_path: Path to the log file
            max_queries: Maximum number of queries to keep in the log
            focus_model: The embedding model to focus on (default: redis-langcache)
        """
        self.log_file_path = log_file_path
        self.max_queries = max_queries
        self.focus_model = focus_model
        self.lock = threading.Lock()
        self.query_history = deque(maxlen=max_queries)
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize the log file if it doesn't exist
        if not os.path.exists(log_file_path):
            self._initialize_log_file()
        else:
            # Load existing statistics
            self._load_stats()

    def _initialize_log_file(self):
        """Initialize the log file with header and structure."""
        with open(self.log_file_path, 'w') as f:
            f.write(f"# Redis Langcache Operation Log ({self.focus_model})\n\n")
            f.write("## Cache Statistics\n\n")
            f.write("- **Total Queries:** 0\n")
            f.write("- **Cache Hits:** 0\n")
            f.write("- **Cache Misses:** 0\n")
            f.write("- **Hit Ratio:** 0.00%\n\n")
            f.write("## Cache Creation Events\n\n")
            f.write("| Timestamp | Cache ID | Model |\n")
            f.write("|-----------|----------|-------|\n")
            f.write("\n## Query History\n\n")
            f.write("| Timestamp | Query | Result | Matched Query | Similarity | Response Time (s) |\n")
            f.write("|-----------|-------|--------|--------------|------------|-------------------|\n")

    def _load_stats(self):
        """Load existing statistics from the log file."""
        try:
            with open(self.log_file_path, 'r') as f:
                content = f.read()

            # Extract hit and miss counts from the file
            hits_line = [line for line in content.split('\n') if "**Cache Hits:**" in line]
            misses_line = [line for line in content.split('\n') if "**Cache Misses:**" in line]

            if hits_line:
                self.cache_hits = int(hits_line[0].split(':')[1].strip())
            if misses_line:
                self.cache_misses = int(misses_line[0].split(':')[1].strip())

            # Load query history (limited to max_queries)
            query_lines = content.split("## Query History\n\n")[1].split("\n")[2:] if "## Query History\n\n" in content else []
            for line in query_lines:
                if line.strip() and "|" in line and not line.startswith("|---"):
                    self.query_history.append(line)
                    if len(self.query_history) >= self.max_queries:
                        break
        except Exception as e:
            print(f"Error loading stats from log file: {e}")
            # If there's an error, reinitialize the log file
            self._initialize_log_file()

    def _update_stats_section(self):
        """Update the statistics section in the log file."""
        total_queries = self.cache_hits + self.cache_misses
        hit_ratio = (self.cache_hits / total_queries) * 100 if total_queries > 0 else 0

        with open(self.log_file_path, 'r') as f:
            content = f.read()

        # Update the stats section
        stats_section = f"## Cache Statistics\n\n"
        stats_section += f"- **Total Queries:** {total_queries}\n"
        stats_section += f"- **Cache Hits:** {self.cache_hits}\n"
        stats_section += f"- **Cache Misses:** {self.cache_misses}\n"
        stats_section += f"- **Hit Ratio:** {hit_ratio:.2f}%\n\n"

        # Replace the existing stats section
        if "## Cache Statistics\n\n" in content:
            parts = content.split("## Cache Statistics\n\n")
            if len(parts) > 1:
                second_part = parts[1].split("## Cache Creation Events\n\n")[1]
                new_content = parts[0] + stats_section + "## Cache Creation Events\n\n" + second_part
                with open(self.log_file_path, 'w') as f:
                    f.write(new_content)

    def log_cache_creation(self, cache_id, model):
        """Log a cache creation event."""
        # Only log cache creation for the focus model
        if model != self.focus_model:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.lock:
            # First, let's check if the file exists and create it if it doesn't
            if not os.path.exists(self.log_file_path):
                self._initialize_log_file()

            # Read the current content
            with open(self.log_file_path, 'r') as f:
                content = f.read()

            # Completely rebuild the cache creation events section
            if "## Cache Creation Events\n\n" in content:
                # Split the content to get the parts before and after the cache events section
                parts = content.split("## Cache Creation Events\n\n")
                header = parts[0] + "## Cache Creation Events\n\n"

                # Get the part after the cache events section
                if "\n## Query History\n\n" in parts[1]:
                    query_history_part = parts[1].split("\n## Query History\n\n")[1]
                    query_history_section = "\n## Query History\n\n" + query_history_part
                else:
                    query_history_section = "\n## Query History\n\n"

                # Create a new cache events table
                cache_events_table = "| Timestamp | Cache ID | Model |\n"
                cache_events_table += "|-----------|----------|-------|\n"

                # Add the new entry
                cache_events_table += f"| {timestamp} | {cache_id} | {model} |\n"

                # Reconstruct the content
                new_content = header + cache_events_table + query_history_section

                # Write the updated content
                with open(self.log_file_path, 'w') as f:
                    f.write(new_content)
            else:
                # If the section doesn't exist, initialize the file
                self._initialize_log_file()

    def log_query(self, query, model, result, similarity=None, response_time=None, matched_query=None):
        """
        Log a query and its result.

        Args:
            query: The query string
            model: The embedding model used
            result: 'hit' or 'miss'
            similarity: Similarity score (for hits)
            response_time: Response time in seconds
            matched_query: The query that was matched in the cache (for hits)
        """
        # Only log queries for the focus model
        if model != self.focus_model:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Truncate query if it's too long
        if len(query) > 40:
            query = query[:37] + "..."

        # Escape pipe characters in the query
        query = query.replace("|", "\\|")

        # Format the similarity and response time
        similarity_str = f"{similarity:.4f}" if similarity is not None else "N/A"
        response_time_str = f"{response_time:.4f}" if response_time is not None else "N/A"

        # Format the matched query
        if matched_query and len(matched_query) > 40:
            matched_query = matched_query[:37] + "..."
        if matched_query:
            matched_query = matched_query.replace("|", "\\|")
        matched_query_str = matched_query if matched_query else "N/A"

        # Create the new entry
        new_entry = f"| {timestamp} | {query} | {result} | {matched_query_str if result.lower() == 'hit' else 'N/A'} | {similarity_str} | {response_time_str} |"

        with self.lock:
            # Update hit/miss counts
            if result.lower() == 'hit':
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            # Update the stats section
            self._update_stats_section()

            # Add to query history
            self.query_history.append(new_entry)

            # Update the query history section in the file
            with open(self.log_file_path, 'r') as f:
                content = f.read()

            if "## Query History\n\n" in content:
                parts = content.split("## Query History\n\n")
                header = parts[0] + "## Query History\n\n"

                # Reconstruct the query history table
                query_table = "| Timestamp | Query | Result | Matched Query | Similarity | Response Time (s) |\n"
                query_table += "|-----------|-------|--------|--------------|------------|-------------------|\n"

                # Add the entries from the deque (most recent first)
                for entry in reversed(self.query_history):
                    query_table += entry + "\n"

                # Write the updated content
                with open(self.log_file_path, 'w') as f:
                    f.write(header + query_table)

# Singleton instance
log_manager = LogManager()
