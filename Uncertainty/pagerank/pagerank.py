import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    N = len(corpus)
    distribution = {}

    # If the page has outgoing links
    if corpus[page]:
        # Probability of random choice among linked pages
        linked_prob = damping_factor / len(corpus[page])
        random_prob = (1 - damping_factor) / N

        for p in corpus:
            if p in corpus[page]:
                distribution[p] = linked_prob + random_prob
            else:
                distribution[p] = random_prob

    # If the page has no outgoing links
    else:
        # Equal probability for all pages
        for p in corpus:
            distribution[p] = 1 / N

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    page_rank = {page: 0 for page in corpus}

    # Start with a random page
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_rank[current_page] += 1
        # Get the next page using transition model
        distribution = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(distribution.keys()), weights=distribution.values())[0]

    # Convert visit counts to probabilities (i.e., PageRank values)
    page_rank = {page: count / n for page, count in page_rank.items()}

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    page_rank = {page: 1 / N for page in corpus}
    new_rank = page_rank.copy()

    while True:
        for page in corpus:
            rank_sum = 0
            for linking_page in corpus:
                if page in corpus[linking_page]:
                    rank_sum += page_rank[linking_page] / len(corpus[linking_page])
                elif not corpus[linking_page]:  # If page has no links
                    rank_sum += page_rank[linking_page] / N

            # Update PageRank using the formula
            new_rank[page] = (1 - damping_factor) / N + damping_factor * rank_sum

        # Check for convergence
        if all(abs(new_rank[page] - page_rank[page]) < 0.001 for page in corpus):
            break

        page_rank = new_rank.copy()

    return page_rank

if __name__ == "__main__":
    main()
