"""
Web scraper untuk collect Laravel-related content dari berbagai sumber
"""
import requests
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict
from urllib.parse import quote_plus


class LaravelDataScraper:
    """Scrape Laravel content dari berbagai sumber"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.data = []

    def scrape_laravel_docs(self, topics: List[str]) -> List[Dict]:
        """
        Scrape Laravel official documentation

        Args:
            topics: List of Laravel topics (e.g., ['eloquent', 'routing', 'migrations'])
        """
        base_url = "https://laravel.com/docs/10.x"
        scraped_data = []

        for topic in topics:
            try:
                url = f"{base_url}/{topic}"
                print(f"Scraping: {url}")

                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Find main content
                    content_div = soup.find('article') or soup.find('div', class_='docs_body')

                    if content_div:
                        # Extract headings and paragraphs
                        sections = []
                        current_heading = None
                        current_content = []

                        for element in content_div.find_all(['h2', 'h3', 'p', 'pre']):
                            if element.name in ['h2', 'h3']:
                                if current_heading:
                                    sections.append({
                                        'heading': current_heading,
                                        'content': ' '.join(current_content)
                                    })
                                current_heading = element.get_text().strip()
                                current_content = []
                            elif element.name == 'p':
                                current_content.append(element.get_text().strip())
                            elif element.name == 'pre':
                                code = element.get_text().strip()
                                if code:
                                    current_content.append(f"Code: {code[:200]}...")

                        # Add last section
                        if current_heading:
                            sections.append({
                                'heading': current_heading,
                                'content': ' '.join(current_content)
                            })

                        for section in sections:
                            scraped_data.append({
                                'source': 'laravel_docs',
                                'topic': topic,
                                'title': section['heading'],
                                'content': section['content'][:1000],  # Limit content length
                                'url': url
                            })

                time.sleep(1)  # Be nice to the server

            except Exception as e:
                print(f"Error scraping {topic}: {e}")
                continue

        self.data.extend(scraped_data)
        return scraped_data

    def search_stackoverflow(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search StackOverflow for Laravel questions
        Note: This is simplified - in production, use StackExchange API
        """
        scraped_data = []

        try:
            # Using Google to search StackOverflow (simple approach)
            search_query = f"site:stackoverflow.com laravel {query}"
            url = f"https://www.google.com/search?q={quote_plus(search_query)}"

            # Note: This might be blocked by Google. Better to use StackExchange API
            print(f"⚠️  For production, use StackExchange API instead")
            print(f"Search query: {search_query}")

            # Placeholder - would need StackExchange API key
            scraped_data.append({
                'source': 'stackoverflow',
                'query': query,
                'note': 'Use StackExchange API for real implementation',
                'api_url': 'https://api.stackexchange.com/docs'
            })

        except Exception as e:
            print(f"Error searching StackOverflow: {e}")

        return scraped_data

    def scrape_laracasts(self, topic: str) -> List[Dict]:
        """
        Scrape Laracasts forum/articles
        Note: Requires proper authentication and API access
        """
        print(f"⚠️  Laracasts requires authentication")
        print(f"Consider using their API or RSS feeds")

        return [{
            'source': 'laracasts',
            'topic': topic,
            'note': 'Requires API access or subscription'
        }]

    def scrape_medium_articles(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Scrape Medium articles about Laravel
        Note: Medium has anti-scraping measures
        """
        scraped_data = []

        try:
            search_url = f"https://medium.com/search?q=laravel%20{quote_plus(query)}"
            print(f"Searching Medium: {query}")

            # Note: Medium blocks scrapers, better to use their API or RSS
            print(f"⚠️  Consider using Medium RSS feeds or API")

            scraped_data.append({
                'source': 'medium',
                'query': query,
                'note': 'Use Medium RSS or API for better results',
                'rss_url': f"https://medium.com/tag/laravel/latest?format=rss"
            })

        except Exception as e:
            print(f"Error scraping Medium: {e}")

        return scraped_data

    def save_scraped_data(self, filepath: str):
        """Save scraped data to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(self.data)} items to {filepath}")

    def load_scraped_data(self, filepath: str):
        """Load previously scraped data"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"✅ Loaded {len(self.data)} items from {filepath}")
        return self.data


# Example usage
if __name__ == "__main__":
    scraper = LaravelDataScraper()

    # Scrape Laravel docs
    topics = ['eloquent', 'routing', 'migrations', 'middleware', 'validation']
    docs_data = scraper.scrape_laravel_docs(topics)

    print(f"\n✅ Scraped {len(docs_data)} sections from Laravel docs")

    # Save data
    scraper.save_scraped_data('./data/raw/scraped_laravel_content.json')

    # Show sample
    if scraper.data:
        print("\n📝 Sample scraped content:")
        print(json.dumps(scraper.data[0], indent=2))
