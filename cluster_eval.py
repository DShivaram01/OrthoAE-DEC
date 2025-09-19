# ortho/cluster_eval.py
import os, time, json, argparse, glob
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from .data_io import load_species_matrix, l2_normalize_rows, compute_k_list_from_tokens

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

    # groups per cluster
    groups_per_cluster = [[] for _ in range(n_clusters)]
    for i_label in range(n_samples):
        groups_per_cluster[X_labels[i_label]].append(prot_meta[i_label][4])

    list_all_groups = list(set([p[4] for p in prot_meta]))

    og_count = np.zeros((len(list_all_groups), n_clusters))
    for c in range(n_clusters):
        for g in groups_per_cluster[c]:
            og_count[list_all_groups.index(g), c] += 1

    if np.sum(og_count) != n_samples:
        print("Warning: count mismatch in group matrix")

    # Family completeness
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

def run_kmeans_eval(
    X_feat: np.ndarray,
    prot_meta: List[List[str]],
    n_init: int,
    max_iter: int,
    algorithm: str,
    seeds: List[int],
    ks_tokens: List,
    out_dir: str,
    variant_tag: str,
):
    N = X_feat.shape[0]
    k_list, ogs = compute_k_list_from_tokens(prot_meta, N, ks_tokens)
    metrics_rows = []

    for seed in seeds:
        for k in k_list:
            t0 = time.time()
            km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, algorithm=algorithm,
                        random_state=seed).fit(X_feat)
            elapsed = time.time() - t0

            saving = saving_from_kmeans(X_feat, prot_meta, km)

            pairs = measure_pairwise_performance(saving, X_feat)
            naive_list, dist_list, one2one_list = pairs

            def count_pairs(pairs_list, idx):
                n_corr, n_tot = 0, len(pairs_list)
                list_all_groups_no_set = [p[4] for p in prot_meta]
                n_species = len(set([p[0] for p in prot_meta]))
                for prots in pairs_list:
                    same_group = (len(set([p[4] for p in prots])) == 1)
                    if not same_group:
                        continue
                    if idx == 2:  # 1:1 check
                        if list_all_groups_no_set.count(prots[0][4]) == n_species:
                            n_corr += 1
                    else:
                        n_corr += 1
                return n_corr, n_tot

            n_corr_naive, n_tot_naive = count_pairs(naive_list, 0)
            n_corr_dist,  n_tot_dist  = count_pairs(dist_list, 1)
            n_corr_121,   n_tot_121   = count_pairs(one2one_list, 2)

            fam, ami, exact = measure_group_performance(saving)

            metrics_rows.append({
                "seed": seed, "k": k, "N": N, "OGs": ogs,
                "naive_correct": n_corr_naive, "naive_total": n_tot_naive,
                "dist_correct":  n_corr_dist,  "dist_total":  n_tot_dist,
                "one2one_correct": n_corr_121, "one2one_total": n_tot_121,
                "family": float(fam), "AMI": float(ami), "exact_pct": float(exact*100.0),
                "kmeans_time_sec": float(elapsed)
            })

    # Save CSV
    import csv
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{variant_tag}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["seed","k","N","OGs",
                     "naive_correct","naive_total","naive_pct",
                     "dist_correct","dist_total","dist_pct",
                     "one2one_correct","one2one_total","one2one_pct",
                     "family","AMI","exact_pct","kmeans_time_sec"])
        for r in metrics_rows:
            cw.writerow([
                r["seed"], r["k"], r["N"], r["OGs"],
                r["naive_correct"], r["naive_total"],
                (100.0*r["naive_correct"]/max(1,r["naive_total"])) if r["naive_total"] else 0.0,
                r["dist_correct"], r["dist_total"],
                (100.0*r["dist_correct"]/max(1,r["dist_total"])) if r["dist_total"] else 0.0,
                r["one2one_correct"], r["one2one_total"],
                (100.0*r["one2one_correct"]/max(1,r["one2one_total"])) if r["one2one_total"] else 0.0,
                r["family"], r["AMI"], r["exact_pct"], r["kmeans_time_sec"]
            ])
    print(f"[saved] {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config")
    ap.add_argument("--exp_dir", required=False, help="Where latents/ids/meta live; if absent, compute from config")
    ap.add_argument("--features", required=False, help="Path to .npy features (optional). If omitted, auto-pick latents in exp_dir.")
    ap.add_argument("--row_l2", action="store_true", help="L2-normalize rows before k-means")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data"]["root"]
    test_spec = cfg["data"]["test_spec"]
    model_name= cfg["data"]["model_name"]
    emb_layer = cfg["data"]["emb_layer"]

    cluster = cfg["cluster"]
    seeds = cluster.get("seeds", [0,1,2])
    n_init= int(cluster.get("n_init", 50))
    max_iter=int(cluster.get("max_iter", 500))
    algo = cluster.get("algo", "elkan")
    ks_tokens = cluster.get("ks", ["N2"])

    # meta from FASTA to count OGs / keep exact parsing
    _, ids, meta = load_species_matrix(test_spec, model_name, emb_layer, data_root)

    if args.features:
        X = np.load(args.features)
        exp_dir = args.exp_dir or os.path.dirname(args.features)
        variant_tag = os.path.splitext(os.path.basename(args.features))[0]
    else:
        # auto pick a latents file in exp_dir
        exp_dir = args.exp_dir
        if not exp_dir:
            raise RuntimeError("--exp_dir required if --features not given")
        cand = sorted(glob.glob(os.path.join(exp_dir, f"{test_spec}_latents_plain_*.npy")))
        if not cand: raise RuntimeError("No *_latents_plain_*.npy found in exp_dir.")
        X = np.load(cand[0])
        variant_tag = os.path.splitext(os.path.basename(cand[0]))[0]

    if args.row_l2:
        X = l2_normalize_rows(X)

    out_dir = os.path.join(exp_dir, "kmeans_eval")
    run_kmeans_eval(
        X_feat=X,
        prot_meta=meta,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algo,
        seeds=seeds,
        ks_tokens=ks_tokens,
        out_dir=out_dir,
        variant_tag=variant_tag,
    )

if __name__ == "__main__":
    main()
