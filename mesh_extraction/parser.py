#!/usr/bin/env python3
"""
Parser for MeSH dataset - loads and processes biomedical articles with MeSH terms
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class MeSHArticle:
    """Data class for a single MeSH article"""
    pmid: str
    title: str
    abstract: str
    mesh_terms: List[str]
    journal: str
    year: str
    
    def __repr__(self):
        return f"MeSHArticle(pmid={self.pmid}, title={self.title[:50]}..., mesh_terms={len(self.mesh_terms)} terms)"

class MeSHParser:
    """Parser for MeSH dataset JSON files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.articles = []
        
    def load_data(self, max_articles: Optional[int] = None, skip: int = 0):
        """
        Load MeSH articles from JSON file
        
        Args:
            max_articles: Maximum number of articles to load (None for all)
            skip: Number of articles to skip from the beginning
        
        Returns:
            List of MeSHArticle objects
        """
        print(f"Loading MeSH data from {self.file_path}")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            print("Warning: UTF-8 decoding failed, using error handling...")
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
        
        raw_articles = data.get('articles', [])
        
        # Apply skip and max_articles
        start_idx = skip
        end_idx = skip + max_articles if max_articles else len(raw_articles)
        selected_articles = raw_articles[start_idx:end_idx]
        
        # Parse into MeSHArticle objects
        self.articles = []
        for article_data in selected_articles:
            # Only include articles with required fields
            if not article_data.get('abstractText') or not article_data.get('meshMajor'):
                continue
                
            article = MeSHArticle(
                pmid=article_data.get('pmid', ''),
                title=article_data.get('title', ''),
                abstract=article_data.get('abstractText', ''),
                mesh_terms=article_data.get('meshMajor', []),
                journal=article_data.get('journal', ''),
                year=article_data.get('year', '')
            )
            self.articles.append(article)
        
        print(f"Loaded {len(self.articles)} valid articles (skipped {skip}, filtered from {len(selected_articles)})")
        return self.articles
    
    def get_article(self, index: int) -> Optional[MeSHArticle]:
        """Get a specific article by index"""
        if 0 <= index < len(self.articles):
            return self.articles[index]
        return None
    
    def get_articles_by_year(self, year: str) -> List[MeSHArticle]:
        """Get all articles from a specific year"""
        return [article for article in self.articles if article.year == year]
    
    def get_articles_by_journal(self, journal: str) -> List[MeSHArticle]:
        """Get all articles from a specific journal"""
        return [article for article in self.articles if journal.lower() in article.journal.lower()]
    
    def get_statistics(self) -> Dict:
        """Get statistics about the loaded dataset"""
        if not self.articles:
            return {}
        
        total_terms = sum(len(article.mesh_terms) for article in self.articles)
        all_unique_terms = set()
        for article in self.articles:
            all_unique_terms.update(article.mesh_terms)
        
        years = [article.year for article in self.articles if article.year]
        journals = [article.journal for article in self.articles if article.journal]
        
        return {
            'total_articles': len(self.articles),
            'total_mesh_terms': total_terms,
            'avg_terms_per_article': total_terms / len(self.articles),
            'unique_mesh_terms': len(all_unique_terms),
            'years': sorted(list(set(years))),
            'unique_journals': len(set(journals)),
            'avg_abstract_length': sum(len(article.abstract) for article in self.articles) / len(self.articles)
        }
    
    def save_subset(self, output_path: str, indices: Optional[List[int]] = None):
        """
        Save a subset of articles to a new JSON file
        
        Args:
            output_path: Path to output file
            indices: List of article indices to save (None for all)
        """
        articles_to_save = self.articles
        if indices:
            articles_to_save = [self.articles[i] for i in indices if 0 <= i < len(self.articles)]
        
        output_data = {
            'articles': [
                {
                    'pmid': article.pmid,
                    'title': article.title,
                    'abstractText': article.abstract,
                    'meshMajor': article.mesh_terms,
                    'journal': article.journal,
                    'year': article.year
                }
                for article in articles_to_save
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(articles_to_save)} articles to {output_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse and explore MeSH dataset')
    parser.add_argument('--file', type=str, required=True, help='Path to MeSH JSON file')
    parser.add_argument('--max', type=int, default=None, help='Maximum articles to load')
    parser.add_argument('--skip', type=int, default=0, help='Articles to skip')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--show', type=int, default=None, help='Show article at index')
    args = parser.parse_args()
    
    # Load data
    mesh_parser = MeSHParser(args.file)
    mesh_parser.load_data(max_articles=args.max, skip=args.skip)
    
    # Show statistics
    if args.stats:
        stats = mesh_parser.get_statistics()
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, list):
                print(f"{key}: {value if len(value) <= 10 else f'{len(value)} items'}")
            elif isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        print("="*60)
    
    # Show specific article
    if args.show is not None:
        article = mesh_parser.get_article(args.show)
        if article:
            print("\n" + "="*60)
            print(f"ARTICLE #{args.show}")
            print("="*60)
            print(f"PMID: {article.pmid}")
            print(f"Title: {article.title}")
            print(f"Journal: {article.journal} ({article.year})")
            print(f"\nAbstract:\n{article.abstract[:500]}...")
            print(f"\nMeSH Terms ({len(article.mesh_terms)}):")
            for term in article.mesh_terms:
                print(f"  - {term}")
            print("="*60)


if __name__ == '__main__':
    main()
