import re
import string
from collections import Counter

# Common English stop words
STOP_WORDS = {
    "a", "an", "the", "and", "or", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "for", "in", "on", "at",
    "by", "with", "from", "as", "that", "this", "these", "those",
    "it", "its", "into", "about", "over", "under", "after",
    "before", "between", "during", "while", "than", "then",
    "if", "else", "not", "can", "could", "should", "would",
    "will", "shall", "do", "does", "did", "have", "has", "had",
    "you", "your", "our", "their", "his", "her", "they", "them",
    "he", "she", "we", "i"
}


def clean_text(text):
    """
    Clean text for similarity calculation.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\S+@\S+", "", text)

    text = re.sub(r"\d+", " ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """
    Split text into tokens.
    """

    text = clean_text(text)

    return text.split()


def remove_stopwords(tokens):
    """
    Remove common stop words.
    """

    return [
        word for word in tokens
        if word not in STOP_WORDS and len(word) > 1
    ]


def preprocess(text):
    """
    Complete preprocessing pipeline.
    """

    tokens = tokenize(text)

    tokens = remove_stopwords(tokens)

    return tokens


def get_word_count(text):
    """
    Return total number of words.
    """

    return len(tokenize(text))


def get_unique_word_count(text):
    """
    Return unique word count.
    """

    return len(set(tokenize(text)))


def extract_keywords(text, top_n=20):
    """
    Return top occurring keywords.
    """

    words = preprocess(text)

    counter = Counter(words)

    return counter.most_common(top_n)


def get_keyword_set(text):
    """
    Return keyword set.
    """

    return set(preprocess(text))


def matching_keywords(resume_text, job_text):
    """
    Common keywords.
    """

    resume = get_keyword_set(resume_text)

    job = get_keyword_set(job_text)

    return sorted(list(resume.intersection(job)))


def missing_keywords(resume_text, job_text):
    """
    Missing keywords.
    """

    resume = get_keyword_set(resume_text)

    job = get_keyword_set(job_text)

    return sorted(list(job - resume))


def resume_statistics(text):
    """
    Return resume statistics.
    """

    words = tokenize(text)

    stats = {
        "total_words": len(words),
        "unique_words": len(set(words)),
        "characters": len(text),
        "keywords": extract_keywords(text, 10)
    }

    return stats
