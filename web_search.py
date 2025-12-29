"""
Web Search Module - Direct web search without LLM
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import time

class WebSearcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def search_duckduckgo(self, query, max_results=5):
        """Search using DuckDuckGo (no API key needed)"""
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='links_main')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if not title_elem:
                    title_elem = result.find('a')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    result_url = title_elem.get('href', '')
                    
                    if not snippet_elem:
                        snippet_elem = result.find('div', class_='result__snippet')
                    
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else 'No description available'
                    
                    if result_url and title:
                        results.append({
                            'title': title,
                            'url': result_url,
                            'snippet': snippet,
                            'source': 'DuckDuckGo'
                        })
            
            return results
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def search_wikipedia(self, query):
        """Search Wikipedia for educational content"""
        try:
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': 3,
                'format': 'json'
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if len(data) >= 4 and isinstance(data, list):
                titles = data[1]
                descriptions = data[2]
                urls = data[3]
                
                for i in range(min(len(titles), len(descriptions), len(urls))):
                    if titles[i] and urls[i]:
                        results.append({
                            'title': titles[i],
                            'url': urls[i],
                            'snippet': descriptions[i] if descriptions[i] else 'Wikipedia article',
                            'source': 'Wikipedia'
                        })
            
            return results
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []
    
    def search_google_fallback(self, query, max_results=5):
        """Fallback: scrape Google search results"""
        try:
            url = f"https://www.google.com/search?q={quote_plus(query)}"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for g in soup.find_all('div', class_='g')[:max_results]:
                title_elem = g.find('h3')
                link_elem = g.find('a')
                snippet_elem = g.find('div', class_=['VwiC3b', 'yXK7lf'])
                
                if title_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    url = link_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else 'No description'
                    
                    if url.startswith('http'):
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': 'Google'
                        })
            
            return results
        except Exception as e:
            print(f"Google search error: {e}")
            return []
    
    def search_simple(self, query):
        """Simple search using Wikipedia summary"""
        try:
            # Get Wikipedia full article
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info',
                'exintro': False,
                'explaintext': True,
                'exchars': 6000,  # Get ~1000 words
                'inprop': 'url',
                'titles': query,
                'redirects': 1
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            results = []
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page in pages.items():
                if page_id != '-1' and 'extract' in page:
                    extract = page.get('extract', '')
                    results.append({
                        'title': page.get('title', query),
                        'url': page.get('fullurl', f'https://en.wikipedia.org/wiki/{query.replace(" ", "_")}'),
                        'snippet': extract,
                        'source': 'Wikipedia'
                    })
            
            return results
        except Exception as e:
            print(f"Simple search error: {e}")
            return []
    
    def search(self, query):
        """Combined search from multiple sources"""
        all_results = []
        
        # Try Wikipedia first for long-form content
        simple_results = self.search_simple(query)
        all_results.extend(simple_results)
        
        # Try Google for comprehensive results
        google_results = self.search_google_fallback(query, max_results=5)
        all_results.extend(google_results)
        
        # Try Wikipedia search API
        wiki_results = self.search_wikipedia(query)
        all_results.extend(wiki_results)
        
        # Try DuckDuckGo for additional sources
        if len(all_results) < 8:
            ddg_results = self.search_duckduckgo(query, max_results=5)
            all_results.extend(ddg_results)
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        
        return unique_results[:8]  # Return top 8 results
    
    def search_images(self, query, max_images=3):
        """Search for images using Google Images"""
        try:
            url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            images = []
            for img in soup.find_all('img')[:max_images + 1]:  # +1 to skip Google logo
                img_url = img.get('src') or img.get('data-src')
                if img_url and img_url.startswith('http'):
                    images.append({
                        'url': img_url,
                        'alt': img.get('alt', query)
                    })
                    if len(images) >= max_images:
                        break
            
            return images
        except Exception as e:
            print(f"Image search error: {e}")
            return []
    
    def format_results(self, results):
        """Format search results for display"""
        if not results:
            return "No results found."
        
        formatted = "🔍 **Web Search Results:**\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"**{i}. {result['title']}**\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   🔗 [{result['source']}]({result['url']})\n"
            
            # Add Google search link for the topic
            if i == 1:
                google_link = f"https://www.google.com/search?q={quote_plus(result['title'])}"
                formatted += f"   🔍 [Search on Google]({google_link})\n"
            
            formatted += "\n"
        
        return formatted
