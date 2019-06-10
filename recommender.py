from vectorizer import get_data, get_vectors
from scipy.spatial.distance import cosine
from collections import Counter
from numpy import mean, sum
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


def recommend_products(
    user_id, purchase_df, vectors, n_products=100, n_similar_users=25
):
    top = get_closest(user_id, vectors, n=n_similar_users)
    top_ids = [k[1] for k in top]
    acc = [k[2] for k in top]
    # print("Recommending with avg. similarity of {}".format(round(mean(acc), 2)))

    products = []
    mini_df = purchase_df.loc[top_ids]
    for product_list in mini_df.tolist():
        products.extend(product_list)

    c = Counter()
    for prod in products:
        c[prod] += 1
    return [
        x[0] for x in sorted(c.items(), key=lambda k: k[1], reverse=True)[:n_products]
    ]


def get_popular_products(product_df, n=100):
    prod_list = []
    for pl in product_df:
        prod_list.extend(pl)
    prod_freq = Counter()
    for p in prod_list:
        prod_freq[p] += 1
    return [
        x[0] for x in sorted(prod_freq.items(), key=lambda k: k[1], reverse=True)[:n]
    ]


def gather_conf_matrix(reco_list, bought_list, product_list):
    pred = sorted(
        [(x, 1) if x in reco_list else (x, 0) for x in product_list], key=lambda k: k[0]
    )
    bought = sorted(
        [(x, 1) if x in bought_list else (x, 0) for x in product_list],
        key=lambda k: k[0],
    )
    tp, tn, fp, fn = 0, 0, 0, 0
    for (prod1, x), (prod2, y) in zip(pred, bought):
        if (x, y) == (0, 0):
            tn += 1
        elif (x, y) == (1, 0):
            fp += 1
        elif (x, y) == (1, 1):
            tp += 1
        elif (x, y) == (0, 1):
            fn += 1
    return tp, tn, fp, fn


def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / sum([tp, tn, fp, fn])
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0
    return accuracy, recall, precision, f1


def test(purchase_df, vectors, limit=100, verbose=False):
    if limit:
        uids = list(vectors)[:limit]
    else:
        uids = list(vectors.keys())

    reco_acc_l, reco_recall_l, reco_prec_l, reco_f1_l = [], [], [], []
    pop_acc_l, pop_recall_l, pop_prec_l, pop_f1_l = [], [], [], []

    all_products = set()
    for p in purchase_df:
        for i in p:
            all_products.add(i)

    ten_per = round(len(uids) / 10)
    for i, user_id in enumerate(uids):
        actual_list = purchase_df.loc[int(user_id)]
        popular_products = get_popular_products(purchase_df, n=len(actual_list))
        reco_list = recommend_products(
            user_id, purchase_df, vectors, n_products=len(actual_list)
        )

        reco_accuracy, reco_recall, reco_precision, reco_f1 = calculate_metrics(
            *gather_conf_matrix(reco_list, actual_list, all_products)
        )
        pop_accuracy, pop_recall, pop_precision, pop_f1 = calculate_metrics(
            *gather_conf_matrix(popular_products, actual_list, all_products)
        )
        reco_acc_l.append(reco_accuracy)
        reco_recall_l.append(reco_recall)
        reco_prec_l.append(reco_prec_l)
        reco_f1_l.append(reco_f1)

        pop_acc_l.append(pop_accuracy)
        pop_recall_l.append(pop_recall)
        pop_prec_l.append(pop_precision)
        pop_prec_l.append(pop_f1)

        if verbose:
            if i % ten_per == 0:
                logging.info("{}% complete".format(round((i * 10) / ten_per)))
            # print("{} matches at {}%".format(pos_algo, (pos_algo * 100) / len(reco_list)))

    print("Mean Recommender Accuracy: {}%".format(round(mean(reco_acc_l), 4) * 100))
    print("Mean Recommender Recall: {}%".format(round(mean(reco_recall_l), 4) * 100))
    print("Mean Recommender Precision: {}%".format(round(mean(reco_prec_l), 4) * 100))
    print("Mean Recommender F1 Score: {}%".format(round(mean(reco_f1_l), 4) * 100))
    print()
    print("Mean Popular Accuracy: {}%".format(round(mean(pop_acc_l), 4) * 100))
    print("Mean Popular Recall: {}%".format(round(mean(pop_recall_l), 4) * 100))
    print("Mean Popular Precision: {}%".format(round(mean(pop_prec_l), 4) * 100))
    print("Mean Popular F1 Score: {}%".format(round(mean(pop_f1_l), 4) * 100))


def main():
    from pprint import pprint
    import matplotlib.pyplot as plt

    purchase_df, user_df = get_data()
    vectors = get_vectors(user_df)
    test(purchase_df, vectors, limit=None, verbose=True)
    # pprint(recommend_products(1000001, purchase_df, vectors))
    # print(vectors[1000001])
    # sim = compare_all(vectors)
    # sim.sort(key=lambda k: k[2], reverse=True)
    # print(sim)


if __name__ == "__main__":
    main()
