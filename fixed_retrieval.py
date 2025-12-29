def retrieve(self, query: str, profile_embedding: np.ndarray, 
            top_k: int = 10, difficulty_filter: Optional[str] = None) -> List[Dict]:
    """Retrieve relevant documents with profile-aware ranking"""
    if self.vectorizer is None or self.doc_matrix is None:
        return []
    
    # Enhanced query processing
    query_normalized = normalize_text(query)
    
    # Add query expansion for better matching
    query_terms = query_normalized.split()
    expanded_terms = []
    
    # Add synonyms and related terms
    term_expansions = {
        'quantum': ['quantum', 'atomic', 'electron', 'orbital', 'numbers'],
        'chemical': ['chemical', 'chemistry', 'bond', 'molecule', 'reaction'],
        'bonding': ['bonding', 'bond', 'ionic', 'covalent', 'molecular'],
        'photosynthesis': ['photosynthesis', 'plant', 'chlorophyll', 'glucose'],
        'calculus': ['calculus', 'derivative', 'integral', 'limit'],
        'newton': ['newton', 'force', 'motion', 'physics', 'law']
    }
    
    for term in query_terms:
        expanded_terms.append(term)
        if term in term_expansions:
            expanded_terms.extend(term_expansions[term])
    
    expanded_query = ' '.join(expanded_terms)
    query_vector = self.vectorizer.transform([expanded_query])
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(query_vector, self.doc_matrix).flatten()
    
    # Boost scores for exact keyword matches
    for i, doc_content in enumerate(self.kb['content']):
        doc_lower = doc_content.lower()
        for term in query_terms:
            if term in doc_lower:
                similarity_scores[i] *= 1.5
    
    # Apply difficulty filter if specified
    if difficulty_filter:
        mask = self.kb['difficulty'] == difficulty_filter
        similarity_scores = similarity_scores * mask.values
    
    # Get top documents with lower threshold
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarity_scores[idx] > 0.05:
            content = self.kb.iloc[idx]['content']
            topic = self._extract_topic_from_content(content, query)
            
            results.append({
                'content': content,
                'source': self.kb.iloc[idx]['source'],
                'topic': topic,
                'difficulty': self.kb.iloc[idx]['difficulty'],
                'score': float(similarity_scores[idx]),
                'index': idx
            })
    
    return results

def _extract_topic_from_content(self, content: str, query: str) -> str:
    """Extract relevant topic from content based on query"""
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Match query terms to content
    if 'quantum' in query_lower and ('quantum' in content_lower or 'electron' in content_lower):
        return 'Quantum Numbers'
    elif 'bond' in query_lower and ('bond' in content_lower or 'ionic' in content_lower or 'covalent' in content_lower):
        return 'Chemical Bonding'
    elif 'photosynthesis' in query_lower and 'photosynthesis' in content_lower:
        return 'Photosynthesis'
    elif 'calculus' in query_lower and ('calculus' in content_lower or 'derivative' in content_lower):
        return 'Calculus'
    elif 'newton' in query_lower and ('newton' in content_lower or 'force' in content_lower):
        return 'Newton\'s Laws'
    
    # Extract from content structure
    parts = content.split('|')
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            if key.strip().lower() in ['topic', 'subject', 'concept']:
                return value.strip()
    
    return 'General Topic'