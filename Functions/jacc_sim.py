def get_jaccard_similarity(users,user1, user2):
    set1 = set(users[user1])
    set2 = set(users[user2])
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union


