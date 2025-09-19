# ortho/eval_metrics.py
from typing import List, Tuple
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

def saving_from_kmeans(X_feat, prot_meta, kmeans):
    n_clusters = kmeans.n_clusters
    n_samples  = X_feat.shape[0]
    X_labels   = kmeans.labels_
    return [n_clusters, n_samples, prot_meta, X_labels, kmeans,
            "n_clusters, n_samples, prot_meta, X_labels, kmeans"]

def measure_pairwise_performance(saved_results, X_feat):
    n_clusters = saved_results[0]
    n_samples  = saved_results[1]
    prot_meta  = saved_results[2]
    X_labels   = saved_results[3]
    kmeans     = saved_results[4]

    X_dist = kmeans.transform(X_feat)**2
    orth_naive, orth_dist, orth_1to1 = [], [], []

    n_species_total = len(set([p[0] for p in prot_meta]))
    for cluster in range(n_clusters):
        ind = [i for i, x in enumerate(X_labels) if x == cluster]
        if len(ind) == 1:
            continue
        ind_sorted = np.argsort(X_dist[ind, cluster])
        all_prots = [prot_meta[ind[i]] for i in ind_sorted]
        all_specs = [prot_meta[ind[i]][0] for i in ind_sorted]

        # naive: all cross-species pairs
        if len(set(all_specs)) > 1:
            for i in range(len(all_specs)-1):
                for j in range(i+1, len(all_specs)):
                    if all_specs[i] != all_specs[j]:
                        orth_naive.append([all_prots[i], all_prots[j]])

        # distance-based: nearest to centroid + nearest of a different species
        j2 = 1
        if all_specs[0] != all_specs[j2]:
            orth_dist.append([all_prots[0], all_prots[j2]])
        else:
            while j2 < len(all_specs):
                j2 += 1
                if j2 < len(all_specs) and all_specs[0] != all_specs[j2]:
                    orth_dist.append([all_prots[0], all_prots[j2]])
                    break

        # 1:1: one per species in cluster
        if len(all_specs) == len(set(all_specs)) == n_species_total:
            orth_1to1.append(all_prots)

    return [orth_naive, orth_dist, orth_1to1]

def measure_group_performance(saved_results):
    n_clusters = saved_results[0]
    n_samples  = saved_results[1]
    prot_meta  = saved_results[2]
    X_labels   = saved_results[3]

    groups_per_cluster = [[] for _ in range(n_clusters)]
    for i_label in range(n_samples):
        groups_per_cluster[X_labels[i_label]].append(prot_meta[i_label][4])

    list_all_groups = list(set([p[4] for p in prot_meta]))

    og_count = np.zeros((len(list_all_groups), n_clusters))
    for c in range(n_clusters):
        for g in groups_per_cluster[c]:
            og_count[list_all_groups.index(g), c] += 1

    # Family completeness (sequence-weighted)
    family = 0.0
    for i in range(len(list_all_groups)):
        family += np.max(og_count[i, :])
    family /= n_samples

    # AMI
    AMI = adjusted_mutual_info_score([p[4] for p in prot_meta], X_labels)

    # Exact OG match %
    success = 0
    for i in range(len(list_all_groups)):
        if np.count_nonzero(og_count[i, :] == 0) == n_clusters - 1:
            c = np.nonzero(og_count[i, :])[0][0]
            if np.count_nonzero(og_count[:, c] == 0) == len(list_all_groups) - 1:
                success += 1
    exact_over_clusters = success / n_clusters if n_clusters > 0 else 0.0
    return [family, AMI, exact_over_clusters]
