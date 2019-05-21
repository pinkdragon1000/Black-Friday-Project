from vectorizer import get_data, get_vectors
from scipy.spatial.distance import cosine
import logging

FORMAT = "[%(asctime)s] - [%(levelname)s] - [%(funcName)s] - %(message)s"
logging.basicConfig(level=20, format=FORMAT)


def get_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def collect_pairs(vectors, limit=1000):
    logging.info("Collecting pairs, limit={}".format(limit))
    pairs = []
    for user1 in vectors.items():
        if limit:
            if len(pairs) > limit:
                break
        for user2 in vectors.items():
            if user1 == user2:
                continue
            else:
                if limit:
                    if len(pairs) > limit:
                        break
                pairs.append((user1, user2))
    logging.info("Pairs collected")
    return pairs


def compare_all(vectors):
    pairs = collect_pairs(vectors, limit=10)
    sim = list(map(comparisons, pairs))
    return sim


def comparisons(comp_tuple):
    (ogid, ogvec), (cmpid, cmpvec) = comp_tuple
    return (ogid, cmpid, get_similarity(ogvec, cmpvec))


def get_closest(user_id, vectors, n=5):
    user_vec = vectors[int(user_id)]
    candidates = [
        ((user_id, user_vec), (k, v)) for k, v in vectors.items() if user_id != k
    ]
    candidates = sorted(
        list(map(comparisons, candidates)), key=lambda k: k[2], reverse=True
    )
    return candidates[:n]


def main():
    from pprint import pprint

    purchase_df, user_df = get_data()
    vectors = get_vectors(user_df)
    pprint(get_closest(1000001, vectors, n=10))
    # print(vectors[1000001])
    # sim = compare_all(vectors)
    # sim.sort(key=lambda k: k[2], reverse=True)
    # print(sim)


if __name__ == "__main__":
    main()
