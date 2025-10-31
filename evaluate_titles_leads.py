#!/usr/bin/env python3
"""
Evaluate title & lead generation with ROUGE-2, ROUGE-L, and BERTScore.

Ground truth columns:   title, lead
Predicted columns:      output_title, output_lead

Per-row metrics + macro averages are produced for:
- Titles: ROUGE-L (F1), ROUGE-2 (F1), BERTScore (F1)
- Leads:  ROUGE-L (F1), ROUGE-2 (F1), BERTScore (F1)

Usage:
  python evaluate_titles_leads.py --csv /path/to/test_articles_outputs.csv \
      --out /path/to/metrics.csv --lang en --lower --stem --rescale

Notes:
- Requires packages: rouge-score, bert-score, torch (for BERTScore).
- If packages are missing, you can pass --auto-install to attempt pip installs.
"""

import argparse
import sys
import subprocess
import pandas as pd
import numpy as np

def maybe_auto_install(pkgs, auto_install=False):
    if not auto_install:
        return
    missing = []
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            missing.append(p)
    if missing:
        print(f"[INFO] Auto-installing missing packages: {missing}")
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        subprocess.check_call(cmd)

def load_metrics_tools(auto_install=False):
    # Ensure dependencies
    maybe_auto_install(["rouge_score"], auto_install=auto_install)
    maybe_auto_install(["bert_score"], auto_install=auto_install)
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    return rouge_scorer, bert_score

def safe_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)

def compute_rouge(rouge_scorer, ref, hyp, use_stemmer=True, lower=True):
    if lower:
        ref, hyp = ref.lower(), hyp.lower()
    scorer = rouge_scorer.RougeScorer(["rouge2","rougeL"], use_stemmer=use_stemmer)
    scores = scorer.score(ref, hyp)
    # Return F1 for both
    r2_f = float(scores["rouge2"].fmeasure)
    rl_f = float(scores["rougeL"].fmeasure)
    return r2_f, rl_f

def compute_bertscore(bert_score, refs, hyps, lang="en", rescale_with_baseline=True, device=None):
    # bert_score expects lists
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F1 = bert_score(cands=hyps, refs=refs, lang=lang, rescale_with_baseline=rescale_with_baseline, device=device)
    return float(F1.mean().item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to input CSV with columns: title, lead, output_title, output_lead")
    parser.add_argument("--out", default="metrics_report.csv", help="Where to save per-row metrics CSV")
    parser.add_argument("--lang", default="en", help="Language code for BERTScore (e.g., 'en', 'vi')")
    parser.add_argument("--lower", action="store_true", help="Lowercase before ROUGE")
    parser.add_argument("--stem", action="store_true", help="Use Porter stemmer in ROUGE")
    parser.add_argument("--no-rescale", dest="rescale", action="store_false", help="Disable BERTScore baseline rescaling")
    parser.add_argument("--auto-install", action="store_true", help="Attempt to pip install missing packages automatically")
    args = parser.parse_args()

    # Load libs (with optional auto-install)
    rouge_scorer, bert_score = load_metrics_tools(auto_install=args.auto_install)

    df = pd.read_csv(args.csv)
    required = ["title","lead","output_title","output_lead"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prepare containers
    title_r2_f, title_rl_f, title_bs_f = [], [], []
    lead_r2_f,  lead_rl_f,  lead_bs_f  = [], [], []

    # Compute per-row metrics
    for _, row in df.iterrows():
        ref_title = safe_text(row["title"])
        hyp_title = safe_text(row["output_title"])
        ref_lead  = safe_text(row["lead"])
        hyp_lead  = safe_text(row["output_lead"])

        # ROUGE for title
        t_r2, t_rl = compute_rouge(rouge_scorer, ref_title, hyp_title, use_stemmer=args.stem, lower=args.lower)
        title_r2_f.append(t_r2); title_rl_f.append(t_rl)

        # ROUGE for lead
        l_r2, l_rl = compute_rouge(rouge_scorer, ref_lead, hyp_lead, use_stemmer=args.stem, lower=args.lower)
        lead_r2_f.append(l_r2); lead_rl_f.append(l_rl)

    # BERTScore (batch for efficiency)
    titles_refs = [safe_text(x) for x in df["title"].tolist()]
    titles_hyps = [safe_text(x) for x in df["output_title"].tolist()]
    leads_refs  = [safe_text(x) for x in df["lead"].tolist()]
    leads_hyps  = [safe_text(x) for x in df["output_lead"].tolist()]

    # Compute BERTScore F1 for titles and leads separately
    title_bs = compute_bertscore(bert_score, titles_refs, titles_hyps, lang=args.lang, rescale_with_baseline=args.rescale)
    lead_bs  = compute_bertscore(bert_score, leads_refs, leads_hyps,   lang=args.lang, rescale_with_baseline=args.rescale)

    # Distribute the averaged BERTScore back to each row (or compute per-row BERTScore if desired)
    # If you'd like per-row BERTScore, uncomment the block below.
    # from bert_score import score as bert_score_fn
    # _, _, title_f1s = bert_score_fn(cands=titles_hyps, refs=titles_refs, lang=args.lang, rescale_with_baseline=args.rescale)
    # _, _, lead_f1s  = bert_score_fn(cands=leads_hyps,  refs=leads_refs,  lang=args.lang, rescale_with_baseline=args.rescale)
    # title_bs_list = [float(x) for x in title_f1s]
    # lead_bs_list  = [float(x) for x in lead_f1s]

    # For simplicity, attach the global BERTScore to each row; comment these two lines if using per-row BERTScore
    title_bs_list = [title_bs] * len(df)
    lead_bs_list  = [lead_bs]  * len(df)

    out = df.copy()
    out["title_rouge2_f1"] = title_r2_f
    out["title_rougeL_f1"] = title_rl_f
    out["title_bertscore_f1"] = title_bs_list

    out["lead_rouge2_f1"] = lead_r2_f
    out["lead_rougeL_f1"] = lead_rl_f
    out["lead_bertscore_f1"] = lead_bs_list

    # Macro averages
    summary = {
        "title_rouge2_f1": float(np.mean(title_r2_f)) if len(title_r2_f) else 0.0,
        "title_rougeL_f1": float(np.mean(title_rl_f)) if len(title_rl_f) else 0.0,
        "title_bertscore_f1": float(title_bs),
        "lead_rouge2_f1": float(np.mean(lead_r2_f)) if len(lead_r2_f) else 0.0,
        "lead_rougeL_f1": float(np.mean(lead_rl_f)) if len(lead_rl_f) else 0.0,
        "lead_bertscore_f1": float(lead_bs),
        "rows": int(len(df))
    }

    out.to_csv(args.out, index=False)
    print("=== Macro Averages ===")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    # Also emit a JSON summary alongside the CSV
    with open(args.out.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved per-row metrics to: {args.out}")
    print(f"Saved summary JSON to: {args.out.replace('.csv', '_summary.json')}")

if __name__ == "__main__":
    main()
