import random

def random_pick_n_index(list1, n):
    """
    Randomly pick n index of a list.

    Args:
        list1 (list): The list to pick from.
        n (int): The number of index to pick.

    Returns:
        list: A list of n randomly picked index.
    """

    if n > len(list1):
        raise ValueError("n must be less than or equal to the length of the list.")

    index_list = []
    while len(index_list) < n:
        index = random.randint(0, len(list1) - 1)
        if index not in index_list:
            index_list.append(index)

    return index_list

def freeze_model_n_layer(model, freeze_layer_list=None):
    # print("model", model)
    # print("model.modules()", model.modules())
    params = list(model.named_parameters())
    # indices = random_pick_n_index(params, n)
    if freeze_layer_list is not None:
        for i in range(len(params)):
            if i in freeze_layer_list:
                params[i][1].requires_grad = False



