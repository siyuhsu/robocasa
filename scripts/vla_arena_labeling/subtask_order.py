"""Robust subtask-order correction for per-segment VLM labels.

Fixes the "completed subtask reappears later" bug WITHOUT the pure-monotonic
carry-forward failure mode (a single spurious early jump to the last subtask
poisoning the whole tail).

Method: weighted Longest Non-Decreasing Subsequence (LNDS) as a "backbone",
then position-fill off-backbone segments.
  - canonical order = first-occurrence index of each subtask in the candidate
    plan (handles cyclic Stage-1 plans).
  - weight each segment by its frame count, so short spurious jumps (forward OR
    backward) are cheap to drop and the longest/heaviest consistent monotonic
    progression wins.
  - segments not on the backbone are relabeled (subtask+reason) to the nearest
    PRECEDING backbone segment; leading off-backbone segments take the first
    backbone label.
Result: a monotonic, outlier-robust per-segment subtask sequence.

NOTE (L1/long-horizon): genuinely repeated subtasks (place A then place B) map
to the SAME canonical index when they share a name; pure non-decreasing keeps
them (OK) but position-fill can mislabel — pass distinct candidate names or use
allow_repeats handling for L1.
"""
from collections import Counter, OrderedDict


def _seg_to_num(overall_segment):
    m = OrderedDict()
    for sid in overall_segment:
        if sid not in m:
            m[sid] = len(m) + 1
    return m


def correct_subtask_order(segment_labels, overall_segment, subtask_list):
    """segment_labels: {seg_num(1-based exec order): [subtask, reason]}.
    overall_segment: per-frame raw segment ids. subtask_list: candidate plan.
    Returns corrected {seg_num: [subtask, reason]}."""
    if not segment_labels:
        return dict(segment_labels)
    order = {}
    for s in subtask_list:
        order.setdefault(s, len(order))
    seg_to_num = _seg_to_num(overall_segment)
    frames = Counter(seg_to_num[sid] for sid in overall_segment)  # seg_num -> #frames

    seg_nums = sorted(segment_labels)                 # execution order
    idx = []
    for sn in seg_nums:
        lab = segment_labels[sn]
        st = lab[0] if isinstance(lab, list) and lab else None
        idx.append(order.get(st))                     # None = unknown/error (never anchors)
    w = [max(1, frames.get(sn, 1)) for sn in seg_nums]
    n = len(seg_nums)

    # weighted longest non-decreasing subsequence over idx (skip None)
    valid = [i for i in range(n) if idx[i] is not None]
    best = {i: w[i] for i in valid}
    parent = {i: -1 for i in valid}
    for a, i in enumerate(valid):
        for j in valid[:a]:
            if idx[j] <= idx[i] and best[j] + w[i] > best[i]:
                best[i] = best[j] + w[i]
                parent[i] = j
    backbone = set()
    if valid:
        end = max(valid, key=lambda i: best[i])
        k = end
        while k != -1:
            backbone.add(k)
            k = parent[k]

    # position-fill: off-backbone -> nearest preceding backbone label
    out = {}
    cur = None
    for i, sn in enumerate(seg_nums):
        if i in backbone:
            cur = segment_labels[sn]
            out[sn] = list(cur)
        else:
            out[sn] = list(cur) if cur is not None else list(segment_labels[sn])
    # leading off-backbone -> first backbone label
    if backbone:
        first_bb = min(backbone)
        first_lab = list(segment_labels[seg_nums[first_bb]])
        for i in range(first_bb):
            out[seg_nums[i]] = list(first_lab)
    return out
