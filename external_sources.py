import re
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

class ExternalSources:
    def __init__(self, google_api_key=None, google_cse_id=None, youtube_api_key=None):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.youtube_api_key = youtube_api_key
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract main keywords using simple NLP"""
        # Remove common question words and stopwords
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 
                     'tell', 'me', 'about', 'explain', 'describe', 'in', 'of', 'to', 'for'}
        
        # Convert to lowercase and split
        words = query.lower().split()
        
        # Filter stopwords and short words
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # If no keywords, use original query
        if not keywords:
            keywords = [query.lower()]
        
        return keywords[:3]  # Return top 3 keywords
    
    def search_pdfs(self, keywords: List[str], max_results=3) -> List[Dict[str, str]]:
        """Search for articles and resources (FREE - no API needed)"""
        try:
            query = ' '.join(keywords)
            # Use DuckDuckGo HTML (more scraping-friendly than Google)
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            resources = []
            
            # DuckDuckGo uses simpler HTML structure
            results = soup.find_all('a', class_='result__a')
            
            for result in results[:max_results]:
                title = result.get_text(strip=True)
                href = result.get('href', '')
                
                if title and href and len(title) > 10:
                    resources.append({
                        'title': title[:100],
                        'url': href,
                        'snippet': f'Resource about {" ".join(keywords)}'
                    })
            
            # Fallback: If DuckDuckGo fails, generate helpful search links
            if not resources:
                resources = [
                    {
                        'title': f'Search "{query}" on Google Scholar',
                        'url': f'https://scholar.google.com/scholar?q={quote_plus(query)}',
                        'snippet': 'Academic papers and research'
                    },
                    {
                        'title': f'Search "{query}" on arXiv',
                        'url': f'https://arxiv.org/search/?query={quote_plus(query)}',
                        'snippet': 'Scientific preprints and papers'
                    },
                    {
                        'title': f'Search "{query}" on Medium',
                        'url': f'https://medium.com/search?q={quote_plus(query)}',
                        'snippet': 'Articles and tutorials'
                    }
                ]
            
            return resources
        
        except Exception as e:
            print(f"Resource search error: {e}")
            # Return helpful search links as fallback
            query = ' '.join(keywords)
            return [
                {
                    'title': f'Search "{query}" on Google Scholar',
                    'url': f'https://scholar.google.com/scholar?q={quote_plus(query)}',
                    'snippet': 'Academic papers and research'
                },
                {
                    'title': f'Search "{query}" on arXiv',
                    'url': f'https://arxiv.org/search/?query={quote_plus(query)}',
                    'snippet': 'Scientific preprints and papers'
                }
            ]
    
    def search_youtube(self, keywords: List[str], max_results=2) -> List[Dict[str, str]]:
        """Search for YouTube videos by scraping (FREE - no API needed)"""
        try:
            # Scrape YouTube directly
            query = ' '.join(keywords)
            url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            # Extract video IDs and titles from page
            video_ids = re.findall(r'"videoId":"([^"]{11})"', response.text)
            titles = re.findall(r'"title":{"runs":\[{"text":"([^"]+)"', response.text)
            channels = re.findall(r'"ownerText":{"runs":\[{"text":"([^"]+)"', response.text)
            
            videos = []
            for i, (vid_id, title) in enumerate(zip(video_ids[:max_results], titles[:max_results])):
                channel = channels[i] if i < len(channels) else 'YouTube'
                videos.append({
                    'title': title,
                    'url': f'https://www.youtube.com/watch?v={vid_id}',
                    'channel': channel
                })
            
            return videos if videos else self._mock_youtube_results(keywords)
        
        except Exception as e:
            print(f"YouTube search error: {e}")
            return self._mock_youtube_results(keywords)
    

    
    def _mock_youtube_results(self, keywords: List[str]) -> List[Dict[str, str]]:
        """Mock YouTube results when API not available"""
        topic = ' '.join(keywords).title()
        return [
            {
                'title': f'{topic} - Introduction',
                'url': f'https://www.youtube.com/watch?v=example1',
                'channel': 'Educational Channel'
            },
            {
                'title': f'What is {topic}?',
                'url': f'https://www.youtube.com/watch?v=example2',
                'channel': 'Learning Hub'
            }
        ]
    
    def get_external_resources(self, query: str) -> Dict:
        """Get all external resources for a query"""
        keywords = self.extract_keywords(query)
        
        pdfs = self.search_pdfs(keywords)
        videos = self.search_youtube(keywords)
        
        return {
            'keywords': keywords,
            'pdfs': pdfs,
            'videos': videos
        }
    
    def format_external_resources(self, resources: Dict) -> str:
        """Format external resources for display"""
        output = []
        
        if resources.get('pdfs'):
            output.append("\n📚 RECOMMENDED ARTICLES & PAPERS:")
            for i, pdf in enumerate(resources['pdfs'], 1):
                output.append(f"  {i}. {pdf['title']}")
                output.append(f"     {pdf['url']}")
        
        if resources.get('videos'):
            output.append("\n🎥 RECOMMENDED YOUTUBE VIDEOS:")
            for i, video in enumerate(resources['videos'], 1):
                output.append(f"  {i}. {video['title']}")
                output.append(f"     {video['url']}")
                output.append(f"     Channel: {video['channel']}")
        
        return '\n'.join(output)
