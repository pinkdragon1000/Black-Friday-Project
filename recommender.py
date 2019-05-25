from vectorizer import get_data, get_vectors
from scipy.spatial.distance import cosine
from collections import Counter
from numpy import mean
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


def recommend_products(user_id, purchase_df, vectors, n_products=100, n_similar_users=25):
    top = get_closest(user_id, vectors, n=n_similar_users)
    top_ids = [k[1] for k in top]
    acc = [k[2] for k in top]
    #print("Recommending with avg. similarity of {}".format(round(mean(acc), 2)))

    products = []
    mini_df = purchase_df.loc[top_ids]
    for product_list in mini_df.tolist():
        products.extend(product_list)

    c = Counter()
    for prod in products:
        c[prod] += 1
    return [x[0] for x in sorted(c.items(), key=lambda k: k[1], reverse=True)[:n_products]]


def get_popular_products(product_df, n=100):
    prod_list = []
    for pl in product_df:
        prod_list.extend(pl)
    prod_freq = Counter()
    for p in prod_list:
        prod_freq[p] += 1
    return [x[0] for x in sorted(prod_freq.items(), key=lambda k: k[1], reverse=True)[:n]]


def test(purchase_df, vectors, limit=100, verbose=False):
    if limit:
        uids = list(vectors)[:limit]
    else:
        uids = list(vectors.keys())

    popular_products = get_popular_products(purchase_df, n=limit)
    success_percentage = []
    popular_percentage = []
    ten_per = round(len(uids) / 10)
    for i, user_id in enumerate(uids):
        reco_list = recommend_products(user_id, purchase_df, vectors)
        actual_list = purchase_df.loc[int(user_id)]

        pos_algo = 0
        pos_rando = 0

        for rec in reco_list:
            if rec in actual_list:
                pos_algo += 1

        for pop in popular_products:
            if pop in actual_list:
                pos_rando += 1

        if verbose:
            if i % ten_per == 0:
                logging.info("{}% complete".format(round((i*10) / ten_per)))
            #print("{} matches at {}%".format(pos_algo, (pos_algo * 100) / len(reco_list)))
        success_percentage.append(pos_algo / len(reco_list))
        popular_percentage.append(pos_rando / len(popular_products))

    print()
    print("Average success rate of algorithm over {} trials: {}%".format(len(uids), round(mean(success_percentage) * 100), 4))
    print("Average success rate of random over {} trials: {}%".format(len(uids), round(mean(popular_percentage) * 100), 4))
    print()



def main():
    from pprint import pprint

    purchase_df, user_df = get_data()
    vectors = get_vectors(user_df)
    test(purchase_df, vectors, limit=None, verbose=True)
    #pprint(recommend_products(1000001, purchase_df, vectors))
    # print(vectors[1000001])
    # sim = compare_all(vectors)
    # sim.sort(key=lambda k: k[2], reverse=True)
    # print(sim)


if __name__ == "__main__":
    main()
