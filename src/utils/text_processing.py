"""
Text processing utilities for the Movie Recommendation System.
"""
# @author 成员 B — 基础推荐算法 & 工具库

import re
from typing import List, Set, Optional
import numpy as np


# Common English stop words
STOP_WORDS: Set[str] = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'and',
    'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
    'that', 'this', 'these', 'those', 'it', 'its', 'he', 'she',
    'his', 'her', 'they', 'their', 'them', 'who', 'which', 'what',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'any'
}


def clean_text(text: str, remove_spaces: bool = True, lowercase: bool = True) -> str:
    """
    Clean text by removing special characters and optionally spaces.

    Parameters
    ----------
    text : str
        Input text to clean
    remove_spaces : bool
        Whether to remove spaces
    lowercase : bool
        Whether to convert to lowercase

    Returns
    -------
    str
        Cleaned text
    """
    if not isinstance(text, str):
        return ''

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove spaces if requested
    if remove_spaces:
        text = text.replace(' ', '')

    return text.strip()


def tokenize(text: str, remove_stopwords: bool = True, min_length: int = 2) -> List[str]:
    """
    Tokenize text into words.

    Parameters
    ----------
    text : str
        Input text to tokenize
    remove_stopwords : bool
        Whether to remove stop words
    min_length : int
        Minimum word length to include

    Returns
    -------
    List[str]
        List of tokens
    """
    if not isinstance(text, str):
        return []

    # Convert to lowercase and split
    words = text.lower().split()

    # Clean each word
    words = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in words]

    # Filter
    tokens = []
    for word in words:
        if len(word) >= min_length:
            if not remove_stopwords or word not in STOP_WORDS:
                tokens.append(word)

    return tokens


def extract_keywords(
    text: str,
    max_keywords: int = 5,
    min_word_length: int = 4
) -> List[str]:
    """
    Extract key words from text (simple frequency-based).

    Parameters
    ----------
    text : str
        Input text
    max_keywords : int
        Maximum number of keywords to return
    min_word_length : int
        Minimum word length for keywords

    Returns
    -------
    List[str]
        List of keywords
    """
    tokens = tokenize(text, remove_stopwords=True, min_length=min_word_length)

    # Count frequencies
    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    return [word for word, _ in sorted_words[:max_keywords]]


def clean_list_text(items: List[str], remove_spaces: bool = True) -> List[str]:
    """
    Clean a list of text items.

    Parameters
    ----------
    items : List[str]
        List of text items
    remove_spaces : bool
        Whether to remove spaces

    Returns
    -------
    List[str]
        List of cleaned text items
    """
    if not isinstance(items, list):
        return []
    return [clean_text(item, remove_spaces=remove_spaces) for item in items if item]


def create_soup(
    genres: List[str],
    director: str,
    cast: List[str],
    keywords: List[str],
    genre_weight: int = 3,
    director_weight: int = 3,
    cast_weight: int = 2,
    keyword_weight: int = 1
) -> str:
    """
    Create a 'soup' of features for similarity calculation.

    Parameters
    ----------
    genres : List[str]
        List of genres
    director : str
        Director name
    cast : List[str]
        List of cast members
    keywords : List[str]
        List of keywords
    genre_weight : int
        Weight multiplier for genres
    director_weight : int
        Weight multiplier for director
    cast_weight : int
        Weight multiplier for cast
    keyword_weight : int
        Weight multiplier for keywords

    Returns
    -------
    str
        Combined feature string
    """
    features = []

    # Add genres with weight
    clean_genres = clean_list_text(genres)
    features.extend(clean_genres * genre_weight)

    # Add director with weight
    if director:
        clean_director = clean_text(director)
        features.extend([clean_director] * director_weight)

    # Add cast with weight
    clean_cast = clean_list_text(cast)
    features.extend(clean_cast * cast_weight)

    # Add keywords with weight
    clean_keywords = clean_list_text(keywords)
    features.extend(clean_keywords * keyword_weight)

    return ' '.join(features)
