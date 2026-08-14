"""
Generate the instructor project-update PDF from results/baselines.json.

    ./.venv/bin/python reports/make_report.py
Produces reports/project_update.pdf. Regenerate any time the baseline numbers change.
"""
import json, os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                ListFlowable, ListItem, HRFlowable)

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = json.load(open(os.path.join(ROOT, "results", "baselines.json")))

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13.5,
                      alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#1a3a5c"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=7, spaceAfter=3,
                    textColor=colors.HexColor("#294d69"))
EQ = ParagraphStyle("eq", parent=BODY, alignment=TA_CENTER, fontName="Helvetica-Oblique",
                    fontSize=10, spaceBefore=4, spaceAfter=6, textColor=colors.HexColor("#111"))
TITLE = ParagraphStyle("title", parent=ss["Title"], fontSize=18, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=ss["Normal"], fontSize=10.5, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444"), spaceAfter=2)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#333"))

def P(t, s=BODY): return Paragraph(t, s)
def bullets(items, s=BODY):
    return ListFlowable([ListItem(Paragraph(i, s), leftIndent=10) for i in items],
                        bulletType="bullet", start="•", leftIndent=12)

story = []

# ------------------------------------------------------------------ header
story += [P("Exploring DPO&rsquo;s Implicit Reward", TITLE),
          P("Preliminary Results &amp; Challenges &mdash; Project Update", SUB),
          P("Graduate Reinforcement Learning &nbsp;|&nbsp; 14 August 2026", SUB),
          Spacer(1, 6), HRFlowable(width="100%", color=colors.HexColor("#c9c9c9")), Spacer(1, 4)]

# ------------------------------------------------------------------ group members (PLACEHOLDER)
story += [P("Group Members", H2)]
members = [["#", "Name", "Student ID", "Email"],
           ["1", "«FILL IN»", "«FILL IN»", "shamima2hossain@gmail.com"],
           ["2", "«FILL IN»", "«FILL IN»", "«FILL IN»"],
           ["3", "«FILL IN»", "«FILL IN»", "«FILL IN»"]]
t = Table(members, colWidths=[0.3*inch, 2.0*inch, 1.4*inch, 2.6*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#294d69")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 8.5), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f6f9")]),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
]))
story += [t, P("<i>Replace &laquo;FILL IN&raquo; before submitting.</i>", SMALL), Spacer(1, 4)]

# ------------------------------------------------------------------ 1. Problem definition + MDP
story += [P("1&nbsp;&nbsp;Problem Definition and MDP Specification", H1)]
story += [P(
    "Direct Preference Optimization (DPO) reparameterizes the RLHF reward in terms of the policy. "
    "Its central claim is that a DPO-trained language model <i>is</i> a reward model via the "
    "<b>implicit reward</b> r&#710;(x,y) = &beta;&middot;log[&pi;<sub>&theta;</sub>(y|x) / &pi;<sub>ref</sub>(y|x)]. "
    "This equivalence is exact only at the optimum and on the training distribution. We ask whether "
    "the implicit reward <b>generalizes as a reward model</b> as well as an explicitly-trained "
    "Bradley&ndash;Terry (BT) reward model, under matched data and compute.")]

story += [P("1.1&nbsp;&nbsp;Token-level MDP", H2)]
story += [P("Autoregressive generation is a finite-horizon, deterministic MDP "
            "M = (S, A, P, R, &rho;<sub>0</sub>, &gamma;):")]
story += [bullets([
    "<b>State</b> s<sub>t</sub> = (x, y<sub>&lt;t</sub>): the prompt x together with the response tokens generated so far.",
    "<b>Action</b> a<sub>t</sub> = y<sub>t</sub> &isin; V: emit the next token from the vocabulary V.",
    "<b>Transition</b> P(s<sub>t+1</sub>|s<sub>t</sub>,a<sub>t</sub>): deterministic append, s<sub>t+1</sub> = (x, y<sub>&le;t</sub>).",
    "<b>Reward</b> R: sparse and terminal &mdash; a scalar r(x,y) delivered at the end-of-sequence token; "
    "the KL-regularized RLHF objective adds a per-token shaping term &minus;&beta;&middot;log[&pi;<sub>&theta;</sub>/&pi;<sub>ref</sub>].",
    "<b>Initial state</b> &rho;<sub>0</sub>: a prompt x &sim; D with empty response.",
    "<b>Horizon / discount</b>: episodic with H = max new tokens, &gamma; = 1. The policy is the LM &pi;<sub>&theta;</sub>(a<sub>t</sub>|s<sub>t</sub>).",
])]

story += [P("1.2&nbsp;&nbsp;Contextual-bandit abstraction (DPO&rsquo;s view)", H2)]
story += [P("DPO collapses the horizon: the context is the prompt x &sim; D, the action is the "
            "<i>entire</i> response y &sim; &pi;(&middot;|x), and the reward is the terminal scalar r(x,y). "
            "The KL-regularized objective is")]
story += [P("max<sub>&pi;</sub> &nbsp; E<sub>x&sim;D, y&sim;&pi;(&middot;|x)</sub>[ r(x,y) ] "
            "&minus; &beta;&middot;D<sub>KL</sub>( &pi;(&middot;|x) &nbsp;||&nbsp; &pi;<sub>ref</sub>(&middot;|x) ).", EQ)]
story += [P("This KL-constrained problem has the closed-form optimum")]
story += [P("&pi;*(y|x) = (1/Z(x)) &middot; &pi;<sub>ref</sub>(y|x) &middot; exp( r(x,y)/&beta; ),", EQ)]
story += [P("which, solved for the reward, gives the reparameterization "
            "r(x,y) = &beta;&middot;log[&pi;*(y|x)/&pi;<sub>ref</sub>(y|x)] + &beta;&middot;log Z(x). "
            "The partition term &beta;&middot;log Z(x) depends on x only, so it cancels in any "
            "same-prompt pairwise comparison.")]

story += [P("1.3&nbsp;&nbsp;Preference model, DPO loss, and the two scorers", H2)]
story += [P("Preferences follow the Bradley&ndash;Terry model "
            "P(y<sub>w</sub> &gt; y<sub>l</sub> | x) = &sigma;( r(x,y<sub>w</sub>) &minus; r(x,y<sub>l</sub>) ), "
            "where y<sub>w</sub> &gt; y<sub>l</sub> denotes that y<sub>w</sub> is preferred to y<sub>l</sub>. "
            "Substituting the reparameterized reward (Z(x) cancels) yields the <b>DPO loss</b>")]
story += [P("L<sub>DPO</sub> = &minus;E<sub>(x,y<sub>w</sub>,y<sub>l</sub>)</sub> log &sigma;( "
            "&beta; log[&pi;<sub>&theta;</sub>(y<sub>w</sub>|x)/&pi;<sub>ref</sub>(y<sub>w</sub>|x)] &minus; "
            "&beta; log[&pi;<sub>&theta;</sub>(y<sub>l</sub>|x)/&pi;<sub>ref</sub>(y<sub>l</sub>|x)] ).", EQ)]
story += [P("<b>Implicit reward (object of study).</b> "
            "r&#710;<sub>&theta;</sub>(x,y) = &beta; log[&pi;<sub>&theta;</sub>(y|x)/&pi;<sub>ref</sub>(y|x)] "
            "= &beta; &Sigma;<sub>t</sub> ( log &pi;<sub>&theta;</sub>(y<sub>t</sub>|x,y<sub>&lt;t</sub>) "
            "&minus; log &pi;<sub>ref</sub>(y<sub>t</sub>|x,y<sub>&lt;t</sub>) ).")]
story += [P("<b>Explicit reward (baseline).</b> A scalar head r<sub>&phi;</sub>(x,y) trained with the "
            "BT loss L<sub>RM</sub> = &minus;E log &sigma;( r<sub>&phi;</sub>(x,y<sub>w</sub>) &minus; r<sub>&phi;</sub>(x,y<sub>l</sub>) ) "
            "on the identical pairs, from the identical SFT initialization.")]
story += [P("<b>Evaluation as a preference classifier.</b> For a held-out pair, predict "
            "y<sub>w</sub> &gt; y<sub>l</sub> iff r(x,y<sub>w</sub>) &gt; r(x,y<sub>l</sub>); the primary metric is "
            "<b>pairwise accuracy</b>. Note ranking is invariant to &beta; (a positive scalar), so &beta; "
            "affects calibration/ECE, not accuracy.")]
story += [P("<b>Research question &amp; hypotheses.</b> Under matched data/compute, how does held-out "
            "preference accuracy of r&#710; compare to r<sub>&phi;</sub>, ID and under shift? "
            "<b>H1:</b> explicit &gt; implicit on held-out ID pairs. "
            "<b>H2:</b> the gap widens under distribution shift. A null result (implicit &asymp; explicit) "
            "would empirically substantiate DPO&rsquo;s claim at small scale.")]

# ------------------------------------------------------------------ 2. Datasets
story += [P("2&nbsp;&nbsp;Benchmark Datasets", H1)]
story += [P("Following the instructor&rsquo;s feedback to use benchmark datasets, all data are "
            "established public benchmarks:")]
ds = [["Dataset", "Role in this study"],
      ["UltraFeedback-binarized\n(HuggingFaceH4/ultrafeedback_binarized)",
       "Training preference pairs and the in-distribution (ID) held-out test set."],
      ["RewardBench\n(allenai/reward-bench)",
       "The standard reward-model evaluation benchmark; its Chat / Chat-Hard / Safety / Reasoning "
       "subsets provide controlled distribution shift (OOD)."],
      ["Anthropic HH-RLHF\n(Anthropic/hh-rlhf)",
       "A classic human-preference benchmark used as a cross-dataset OOD set."]]
t2 = Table(ds, colWidths=[2.3*inch, 4.0*inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#294d69")), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.5),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f6f9")]),
    ("VALIGN",(0,0),(-1,-1),"TOP"), ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),
]))
story += [t2, Spacer(1, 3)]

# ------------------------------------------------------------------ 3. Baselines + results
story += [P("3&nbsp;&nbsp;Initial Baselines: Implementations and Results", H1)]
story += [P("3.1&nbsp;&nbsp;Reference baselines (implemented and run)", H2)]
story += [P("These non-neural baselines are the reference floors the neural reward models must beat. "
            "They require no GPU and are already evaluated on every benchmark set:")]
story += [bullets([
    "<b>Random</b> &mdash; chance (0.500).",
    "<b>Pick-longer / Pick-shorter</b> &mdash; predict the longer (resp. shorter) response wins; "
    "probes the length confound flagged in the proposal.",
    "<b>TF-IDF + Logistic Regression</b> &mdash; a learned lexical scorer: TF-IDF over (prompt, response), "
    "trained on UltraFeedback to score a response chosen/rejected, higher-scoring side wins.",
])]

# results table from JSON
res = RESULTS["results"]
order = ["UltraFeedback (ID)", "HH-RLHF harmless (OOD)", "RewardBench: Chat (OOD)",
         "RewardBench: Chat-Hard (OOD)", "RewardBench: Safety (OOD)",
         "RewardBench: Reasoning (OOD)", "RewardBench: ALL (OOD)"]
header = ["Test set", "n", "Random", "Pick-longer", "Pick-shorter", "TF-IDF+LR"]
rows = [header]
for k in order:
    if k not in res: continue
    r = res[k]
    rows.append([k, str(r["n"]), f"{r['random']:.3f}", f"{r['pick_longer']:.3f}",
                 f"{r['pick_shorter']:.3f}", f"{r['tfidf_lr']:.3f}"])
t3 = Table(rows, colWidths=[2.35*inch, 0.5*inch, 0.7*inch, 0.85*inch, 0.85*inch, 0.85*inch])
t3.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#294d69")), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.3),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f6f9")]),
    ("ALIGN",(1,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("TOPPADDING",(0,0),(-1,-1),2.5),("BOTTOMPADDING",(0,0),(-1,-1),2.5),
    # highlight the striking length-bias cell (RewardBench Chat, pick-longer)
    ("TEXTCOLOR",(3,3),(3,3),colors.HexColor("#b00020")), ("FONTNAME",(3,3),(3,3),"Helvetica-Bold"),
]))
story += [t3, P(f"<i>Pairwise accuracy. TF-IDF+LR trained on {RESULTS['train_size']:,} "
                "UltraFeedback pairs (seed 0). Ties count as 0.5.</i>", SMALL)]

story += [P("3.2&nbsp;&nbsp;Key preliminary findings", H2)]
story += [bullets([
    "<b>Length bias is large and dataset-dependent.</b> Pick-longer reaches "
    f"{res['RewardBench: Chat (OOD)']['pick_longer']:.3f} on RewardBench-Chat but only "
    f"{res['RewardBench: Chat-Hard (OOD)']['pick_longer']:.3f} on the adversarial Chat-Hard "
    "(which deliberately inverts length). This validates the proposal&rsquo;s length-controlled "
    "metric as essential &mdash; a scorer can look strong purely by proxying length.",
    "<b>A lexical learner transfers partially.</b> TF-IDF+LR beats chance in-distribution "
    f"({res['UltraFeedback (ID)']['tfidf_lr']:.3f}) and on easy Chat "
    f"({res['RewardBench: Chat (OOD)']['tfidf_lr']:.3f}), but collapses to near/below chance on "
    f"Chat-Hard ({res['RewardBench: Chat-Hard (OOD)']['tfidf_lr']:.3f}) and Reasoning "
    f"({res['RewardBench: Reasoning (OOD)']['tfidf_lr']:.3f}) &mdash; quantifying how much of the "
    "signal is shallow lexical overlap and setting a floor for the neural models.",
    "The ID&rarr;OOD accuracy drop already visible here previews exactly the H2 comparison we will make "
    "between the implicit and explicit rewards.",
])]

story += [P("3.3&nbsp;&nbsp;Neural conditions (implemented, queued for GPU)", H2)]
story += [P("The full training and evaluation pipeline is implemented and syntax-verified; it targets "
            "Qwen2.5-0.5B with LoRA via HuggingFace TRL on a single GPU (Colab). Three neural scorers "
            "are ready to run:")]
story += [bullets([
    "<b>DPO implicit reward</b> r&#710; &mdash; extracted as the &beta;-scaled response log-ratio "
    "(<i>src/scoring.py</i>), to be verified against DPOTrainer&rsquo;s own logged rewards.",
    "<b>Explicit Bradley&ndash;Terry RM</b> r<sub>&phi;</sub> &mdash; scalar head, matched to DPO "
    "(same SFT init, pairs, epochs, LoRA, batch, LR; see <i>DESIGN.md</i>).",
    "<b>Base-model log-probability</b> &mdash; a free neural baseline (length-normalized).",
])]
story += [P("Evaluation reports pairwise accuracy (primary), ECE, length-controlled accuracy, Spearman "
            "vs gold, and a McNemar paired test between the implicit and explicit scorers "
            "(<i>src/metrics.py</i>, <i>src/eval.py</i>).")]

# ------------------------------------------------------------------ 4. Challenges
story += [P("4&nbsp;&nbsp;Challenges", H1)]
story += [bullets([
    "<b>Auditable budget-matching.</b> The comparison is only meaningful if DPO and the RM see "
    "identical data/compute. We fix a shared SFT initialization and match epochs, LoRA config, batch, "
    "and LR, and log tokens-seen / wall-clock &mdash; but defining &lsquo;equal compute&rsquo; across two "
    "different losses (DPO&rsquo;s extra reference forward pass) remains a judgment call we must defend.",
    "<b>Correct implicit-reward extraction.</b> The &beta;-scaled, prompt-masked, length-summed log-ratio "
    "must exactly match TRL&rsquo;s internal convention; a subtle masking or scaling error would silently "
    "bias every downstream number. Mitigation: cross-check against DPOTrainer&rsquo;s logged rewards.",
    "<b>Small-model noise.</b> At 0.5B, reward accuracy can sit close to chance on hard subsets "
    "(the TF-IDF collapse on Reasoning previews this), so headline claims need &ge;3 seeds with "
    "confidence intervals; we may escalate to 1.5B if the signal is too noisy.",
    "<b>Length as a confound.</b> Both scorers can proxy length (Section 3.2); isolating true preference "
    "signal requires the length-controlled metric and reporting the length-only baseline everywhere.",
    "<b>Gold-judge / Spearman set on limited compute.</b> A larger instruction-tuned judge for scalar "
    "gold ratings must be run as a separate quantized/offline pass (or via API) to fit a single "
    "consumer GPU alongside training.",
    "<b>Toolchain drift.</b> TRL&rsquo;s DPOTrainer / RewardTrainer APIs shift across releases; versions "
    "are pinned in <i>requirements.txt</i> and the trainers may need minor version-specific edits on Colab.",
])]

# ------------------------------------------------------------------ 5. Next steps
story += [P("5&nbsp;&nbsp;Next Steps", H1)]
story += [bullets([
    "Run SFT &rarr; DPO &rarr; BT-RM end-to-end at the 8k budget on Colab; produce the first "
    "H1/H2 numbers (implicit vs explicit, ID + OOD) with the reference floors above as context.",
    "Widen to the ablations: &beta; &isin; {0.05, 0.1, 0.3, 0.5}, budget &isin; {2k, 8k, 32k}, and the "
    "training-progress checkpoint curve (overoptimization).",
    "Add the gold-judge OOD set and Spearman correlation; finalize with &ge;3 seeds and significance tests.",
])]
story += [Spacer(1, 4), HRFlowable(width="100%", color=colors.HexColor("#c9c9c9")),
          P("Code, benchmark data pipeline, and reproducible baseline results: "
            "github.com/silvererudite/dpo-implicit-reward", SMALL)]

# ------------------------------------------------------------------ build
out = os.path.join(ROOT, "reports", "project_update.pdf")
SimpleDocTemplate(out, pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch,
                  leftMargin=0.8*inch, rightMargin=0.8*inch,
                  title="Exploring DPO's Implicit Reward - Project Update").build(story)
print("Wrote", out)
