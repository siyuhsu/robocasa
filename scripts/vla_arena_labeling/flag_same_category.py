#!/usr/bin/env python3
"""Flag same-category multi-object episodes for manual review.

Pure LNDS is reliable for distinct-object instructions (lime+banana, tomato+
orange) but can mislabel order/identity when the instruction references >=2
VISUALLY-SIMILAR same-category objects (two mugs, all the apples), because the
VLM can't tell them apart. This emits a review list of such episodes.

Heuristics (over-inclusive on purpose — it's a review list):
  - quantifier: all / both / two / three / second / remaining / other / each
  - a manipulated-object noun repeated >=2x in the instruction (e.g. mug...mug)
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

QUANT = re.compile(r"\b(all|both|two|three|four|second|third|fourth|remaining|another|other|each|every)\b", re.I)
# words that are not manipulated-object nouns (verbs, preps, articles, common destinations)
STOP = set("the a an and or on in to of it them its up then with from out onto into at by "
           "put pick place move moves grasp lift push slide take pull open close set drop release reach carry "
           "left right plate table counter top side region between cabinet drawer shelf sink stove microwave "
           "is are be that this these those your you robot arm gripper hand".split())


# landmarks / destinations / surfaces — repeated mention is NOT a same-category
# manipulation (e.g. "tomato next to the cutting board ... on the cutting board")
SURFACE = set("board plate table counter basket stove cabinet shelf sink microwave region "
              "box bin tray rack pan cutting teapot teapots".split())
SINGULAR_S = set("scissors glasses tongs pliers headphones pants bus".split())  # end in 's' but singular


def reasons_for(instr):
    """Return (reasons, confidence). high = likely same-category (VLM can't tell
    the objects apart); low = quantifier present but objects look distinct."""
    low = instr.lower()
    words = re.findall(r"[a-z]+", low)
    content = [w for w in words if w not in STOP and len(w) > 2]
    cnt = Counter(content)
    # landmark context: "between/next to/near the X" — X is a reference, not a target
    landmark = set(re.findall(r"(?:between|next to|beside|near) the ([a-z]+)", low))
    r = []
    conf = "low"
    # plural manipulated object => multiple same-category (apples, moka pots)
    plurals = sorted({w for w in content if w.endswith("s") and len(w) >= 4
                      and w not in SINGULAR_S and w not in SURFACE and w[:-1] not in SURFACE
                      and w not in landmark})
    if plurals:
        r.append("plural:" + ",".join(plurals)); conf = "high"
    # same head-noun repeated (mug ... mug), excluding landmarks/surfaces
    rep = sorted({w for w, c in cnt.items() if c >= 2 and w not in SURFACE and w not in landmark})
    if rep:
        r.append("repeated-noun:" + ",".join(rep)); conf = "high"
    # quantifier alone (e.g. "both A and B") — often DISTINCT objects -> low/check
    if QUANT.search(low) and not r:
        r.append("quantifier(check-distinct?)")
    return r, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", nargs="+", required=True, help="lerobot suite dirs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    review = {}
    for lr in a.lerobot:
        suite = Path(lr).name.replace("_no_noops_1.0.0_lerobot", "")
        by_instr = defaultdict(list)
        ep = Path(lr) / "meta" / "episodes.jsonl"
        if not ep.exists():
            print(f"[skip] {suite}: no episodes.jsonl"); continue
        for line in open(ep):
            r = json.loads(line); instr = (r.get("tasks") or [""])[0]
            by_instr[instr].append(r["episode_index"])
        flagged = {}
        for instr, eps in by_instr.items():
            rs, conf = reasons_for(instr)
            if rs:
                flagged[instr] = {"reasons": rs, "confidence": conf, "n": len(eps), "episodes": sorted(eps)}
        hi = {k: v for k, v in flagged.items() if v["confidence"] == "high"}
        n_ep_hi = sum(v["n"] for v in hi.values())
        review[suite] = {"n_high": len(hi), "n_episodes_high": n_ep_hi,
                         "n_low": len(flagged) - len(hi), "total_instructions": len(by_instr), "flagged": flagged}
        print(f"=== {suite}: HIGH={len(hi)} instr ({n_ep_hi} ep) | low/check={len(flagged)-len(hi)} | of {len(by_instr)} ===")
        for instr, v in sorted(flagged.items(), key=lambda x: x[1]["confidence"]):
            print(f"   [{v['confidence']:>4}|{v['n']:>3}ep] {v['reasons']}  {instr[:58]}")
    json.dump(review, open(a.out, "w"), indent=2)
    print(f"\n[review list] -> {a.out}")


if __name__ == "__main__":
    main()
