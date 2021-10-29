# coding = utf8

from diffnet.data.utils import read_graph


def generate_user_item_graph(order):

    user_user_graph = read_graph("/home/caidesheng/social-rec/code/diffnet/data/tmp/{}_user_user_graph.pkl".format(order))
    item_item_graph = read_graph("/home/caidesheng/social-rec/code/diffnet/data/tmp/{}_item_item_graph.pkl".format(order))
    user_user_social_graph = read_graph("/home/caidesheng/social-rec/code/diffnet/data/tmp/{}_user_user_social_graph.pkl".format(order))

    user_user_graph = user_user_graph.maximum(user_user_graph.T)
    item_item_graph = item_item_graph.maximum(item_item_graph.T)
    user_user_social_graph = user_user_social_graph.maximum(user_user_social_graph.T)

    def compute_rate(A):
        return len(A.data) / (A.shape[0] * A.shape[1] * 1.)

    print("uu rate: ", compute_rate(user_user_graph))
    print("ii rate: ", compute_rate(item_item_graph))
    print("uus rate: ", compute_rate(user_user_social_graph))

    # save_graph()

    print()
    return user_user_graph, item_item_graph, user_user_social_graph