def get_value_similarity(test_dict : dict, candidate_dict : dict) -> float:
    """
    Get the similarity between two dictionaries. The similarity is calculated as the proportion of values that are in common between the two dictionaries.
    """
    
    score = 0
    total = sum(test_dict.values())
    
    for key in test_dict.keys():
        if key in candidate_dict:
            score += candidate_dict[key]
    
    return score / total
    
    
def get_key_similarity(test_dict : dict, candidate_dict : dict) -> float:
    """
    Get the similarity between two dictionaries. The similarity is calculated as the proportion of keys that are in common between the two dictionaries.
    """
    test_keys = set(test_dict.keys())
    candidate_keys = set(candidate_dict.keys())
    common_keys = test_keys.intersection(candidate_keys)
    similarity = len(common_keys) / len(test_keys)
    return similarity


def get_most_likely_candidates(candidate_lists:list)->str:
    """
    Get the most likely candidate from a list of candidates. The most likely candidate is the one wich appears the most times in each list
    """
    candidate_dict = {}
    
    for candidates in candidate_lists:
        for candidate in candidates:
            if candidate in candidate_dict:
                candidate_dict[candidate] += 1
            else:
                candidate_dict[candidate] = 1
                
    return candidate_dict

def get_jaccard_score(inferred_candidates:dict, test_candidates:dict)->float:
    """
    Get the jaccard score of the inferred candidates. The score is calculated as the Jaccard similarity between the inferred candidates and the test candidates.
    """
    set1 = set(inferred_candidates.keys())
    set2 = set(test_candidates.keys())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union)
    return similarity

def cosine_similarity(dict1, dict2):
    """
    Compute the cosine similarity between two dictionaries. The dictionaries should have the same keys.
    """
    vec1 = [dict1[word] for word in dict1.keys()]
    vec2 = [dict2[word] for word in dict2.keys()]
    dot_product = sum(x * y for x, y in zip(vec1, vec2))

    magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude2 = sum(y ** 2 for y in vec2) ** 0.5

    if magnitude1 != 0 and magnitude2 != 0:
        similarity = dot_product / (magnitude1 * magnitude2)
    else:
        similarity = 0

    return similarity