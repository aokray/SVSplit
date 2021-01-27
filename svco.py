import numpy as np
from scipy.optimize import minimize
from operator import itemgetter
from collections import Counter


def lin_pred(model_coefs):
    X = unlabelled[:, 0:-1]
    y_hat = X.dot(model_coefs)

    y_hat[np.where(y_hat < 0)] = -1
    y_hat[np.where(y_hat >= 0)] = 1

    y_hat = y_hat.reshape(X.shape[0], 1)

    return y_hat


def confidence(model_coefs, preds):
    X = unlabelled[:, 0:-1]

    xdotmodel = X.dot(model_coefs)

    exp_val = -(xdotmodel * preds)

    one_plus = 1 + np.exp(exp_val)

    confs = (one_plus) ** (-1)

    return confs


def obj(args):
    args = args.reshape(p, 2)

    u = args[:, 0]
    v = args[:, 1]

    samp = labelled[:, 0:-2]
    # Using predicted labels whenever possible, we are under assumption that true labels are unknown
    lab = labelled[:, -1].reshape(len(labelled[:, -1]), 1)

    u_side = np.array(
        [np.log(1 + np.exp((-u.T.dot(x) * y)[0])) for (x, y) in zip(samp, lab)]
    )
    v_side = np.array(
        [np.log(1 + np.exp((-v.T.dot(x) * y)[0])) for (x, y) in zip(samp, lab)]
    )

    u_sum = np.sum(u_side)
    v_sum = np.sum(v_side)

    return np.log(v_sum + u_sum)


def constraint1(args):
    args = args.reshape((p, 2))

    u = args[:, 0]
    v = args[:, 1]

    return (u ** 2).T.dot(v ** 2)


def constraint2(args):
    # Make sure we're modifying global version
    global labelled, unlabelled

    args = args.reshape(p, 2)

    u = args[:, 0].reshape(p, 1)
    v = args[:, 1].reshape(p, 1)

    # Calculate confidence and predictions based on Sec 3.3
    u_preds = lin_pred(u)
    v_preds = lin_pred(v)

    cu = confidence(u, u_preds)
    cv = confidence(v, v_preds)

    cu_order = np.argsort(cu.flatten())[::-1]
    cv_order = np.argsort(cv.flatten())[::-1]

    cu_cand_idxs = np.where(cu[cu_order] >= conf_threshold)[0]
    cv_cand_idxs = np.where(cv[cv_order] >= conf_threshold)[0]

    conf_dict = {}

    if len(cu_cand_idxs) > 0:
        for c_idx, conf in zip(
            cu_order[cu_cand_idxs],
            cu[cu_order[cu_cand_idxs]],  # .reshape(v_preds.shape[0], 1),
        ):
            conf_dict[c_idx] = [conf[0], u_preds[c_idx][0]]

    if len(cv_cand_idxs) > 0:
        for c_idx, conf in zip(
            cv_order[cv_cand_idxs],
            cu[cu_order[cv_cand_idxs]],  # .reshape(v_preds.shape[0], 1),
        ):
            if c_idx in conf_dict:
                if conf_dict[c_idx][0] < conf[0]:
                    conf_dict[c_idx] = [conf[0], v_preds[c_idx][0]]
            else:
                conf_dict[c_idx] = [conf[0], v_preds[c_idx][0]]

    if len(cu_cand_idxs) > 0 or len(cv_cand_idxs) > 0:
        k = Counter(conf_dict)
        result = k.most_common(l)

        # Get the indexes of the most confident l inputs from the unlabelled set
        l_high_idxs = np.array(result)[:, 0]
        l_high_idxs = l_high_idxs.astype(int)

        labs = np.array(
            [lab for lab in map(itemgetter(1), np.array(result)[:, 1])]
        ).flatten()

        for i in range(len(labs)):
            if type(labs[i]) == np.ndarray:
                labs[i] = i

        labs = labs.reshape(len(l_high_idxs), 1)

        assert np.all(labs <= 1)

        to_add = unlabelled[l_high_idxs]

        # Add the predicted label to the end of the matrix
        to_add = np.append(to_add, labs, axis=1)

        # Officially add UP TO the l most confident instances to the labelled set
        labelled = np.append(labelled, to_add, axis=0)
        unlabelled = np.delete(unlabelled, l_high_idxs, axis=0)
        print(
            "New labelled set size after adding ",
            str(len(to_add)),
            " instance(s): ",
            str(labelled.shape),
        )

    non_cu = 1 - cu
    non_cv = 1 - cv

    only_one_conf = np.sum((cu * non_cv) + (non_cu + cv))
    both_conf = np.sum((cu * cv))
    neither_conf = np.sum((non_cu * non_cv))
    both_or_neither_conf = epsilon * np.minimum(both_conf, neither_conf)

    return only_one_conf - both_or_neither_conf


def constraint3(args):
    args = args.reshape((p, 2))

    u = args[:, 0]
    v = args[:, 1]

    return (np.sum(u) + np.sum(v)) - p


# Constraint2 >= 0
cons = [
    {"type": "eq", "fun": constraint1},
    {"type": "ineq", "fun": constraint2},
    {"type": "eq", "fun": constraint3},
]


def SVCoTrain(data, ct, eps, to_add):
    global labelled, unlabelled, n, p, epsilon, conf_threshold, l

    sample = data[:, 0:-1]

    (n, p) = sample.shape

    # As per paper's initialization
    conf_threshold = ct
    epsilon = eps
    # l is max # of unlabelled instance to move to labelled instances
    l = to_add

    # Randomly initialize in [0,1] two coef. vectors
    u = np.random.rand(p).reshape(p, 1)
    v = np.random.rand(p).reshape(p, 1)
    both = np.append(u, v, axis=1)
    both = both.flatten()

    percent_label = 0.2

    bnds = [(0, 1)] * 2 * p

    l_idxs = np.random.choice(
        range(sample.shape[0]), int(sample.shape[0] * percent_label), replace=False
    )
    u_idxs = np.setdiff1d(np.array(range(sample.shape[0])), l_idxs)

    # LABELLED FORMAT:
    # SAMPLE + TRUE LABEL + PREDICTED LABEL
    # For initially labelled set, True label = Predicted label

    labelled = data[l_idxs]
    labelled = np.append(
        labelled, labelled[:, -1].reshape(len(labelled[:, -1]), 1), axis=1
    )
    print("Initial Labelled Set Size: ", labelled.shape)

    # UNLABELLED FORMAT:
    # SAMPLE + TRUE LABEL
    unlabelled = data[u_idxs]

    opt = minimize(
        obj, both, constraints=cons, method="SLSQP", bounds=bnds, tol=0.00001
    )

    print(opt)

    print("FINAL LABELLED SHAPE", labelled.shape)

    feat_votes = np.round(opt.x.reshape(p, 2))

    v1_idxs = np.where(feat_votes[:, 0] == 1)[0]
    v2_idxs = np.where(feat_votes[:, 1] == 1)[0]

    return v1_idxs, v2_idxs, labelled
