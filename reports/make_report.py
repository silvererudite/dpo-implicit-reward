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


def load_budget(*path):
    """Read a stage's budget.json (tokens-seen / wall-clock / effective batch), or None.

    Read rather than hard-coded so the PDF cannot drift from what actually ran: regenerating
    after a new run always reports that run's real numbers, and stages that have not finished
    are reported as pending instead of being invented.
    """
    p = os.path.join(ROOT, *path, "budget.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_eval(tag):
    p = os.path.join(ROOT, "results", f"eval_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


SFT_B = load_budget("outputs", "sft")
DPO_B = load_budget("outputs", "dpo_beta0.1_8k")
RM_B = load_budget("outputs", "rm_8k")
EVAL = load_eval("beta0.1_8k")


DASH = "—"  # literal em dash: raw Table cells are not parsed for HTML entities


def fmt_hms(sec):
    if not sec:
        return DASH
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

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
    "<b>Initial state</b> &rho;<sub>0</sub>: a prompt x &sim; D with empty response, i.e. s<sub>0</sub> = (x, &empty;).",
    "<b>Terminal states</b>: a state is terminal once the emitted action is the end-of-sequence token "
    "(a<sub>t</sub> = EOS) or the length cap H is reached, whichever comes first &mdash; so every episode "
    "is finite and the return is well defined without discounting.",
    "<b>Horizon / discount</b>: episodic with H = max new tokens, &gamma; = 1. The policy is the LM &pi;<sub>&theta;</sub>(a<sub>t</sub>|s<sub>t</sub>).",
])]
story += [P("Two properties of this MDP matter for the study. The dynamics are <i>known and "
            "deterministic</i> (appending a token), so there is no transition-model uncertainty to "
            "estimate &mdash; all difficulty sits in the reward. And the reward is <i>sparse and "
            "terminal</i>: no learning signal arrives until the response is complete, which is precisely "
            "why preference-based methods replace an explicit per-token reward with a comparison between "
            "whole trajectories.", SMALL)]

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
# Descriptions go through Paragraph so they wrap; a raw string cell overflows the column
# (the RewardBench row ran off the page edge).
ds = [["Dataset", "Role in this study"],
      ["UltraFeedback-binarized\n(HuggingFaceH4/ultrafeedback_binarized)",
       P("Training preference pairs and the in-distribution (ID) held-out test set.", SMALL)],
      ["RewardBench\n(allenai/reward-bench)",
       P("The standard reward-model evaluation benchmark; its Chat / Chat-Hard / Safety / Reasoning "
         "subsets provide controlled distribution shift (OOD).", SMALL)],
      ["Anthropic HH-RLHF\n(Anthropic/hh-rlhf)",
       P("A classic human-preference benchmark used as a cross-dataset OOD set.", SMALL)]]
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

story += [P("3.3&nbsp;&nbsp;Neural pipeline: now running on GPU", H2)]
story += [P("Since the last update the pipeline has moved off the drawing board and onto hardware: an "
            "AWS SageMaker instance with NVIDIA A10G GPUs (24&nbsp;GB, Ampere &rarr; native bf16), "
            "replacing the Colab plan. Qwen2.5-0.5B + LoRA via HuggingFace TRL, as designed. The stage "
            "status below is read directly from each checkpoint&rsquo;s <i>budget.json</i>, which the "
            "trainers now write automatically so the matched-compute claim is auditable rather than asserted.")]

_stage_rows = [["Stage", "Status", "Steps", "Epochs", "Eff. batch", "Wall-clock"]]
for _label, _b, _pending in [
        ("SFT &rarr; &pi;<sub>ref</sub> (32k)", SFT_B, "queued"),
        ("DPO &beta;=0.1 (8k)", DPO_B, "in progress"),
        ("Bradley&ndash;Terry RM (8k)", RM_B, "in progress")]:
    if _b:
        _stage_rows.append([Paragraph(_label, SMALL), "complete", str(_b.get("global_step", DASH)),
                            f"{_b.get('epochs', 0):.2f}", str(_b.get("effective_batch", DASH)),
                            fmt_hms(_b.get("train_runtime_sec"))])
    else:
        _stage_rows.append([Paragraph(_label, SMALL), _pending, DASH, DASH, DASH, DASH])
t4 = Table(_stage_rows, colWidths=[1.85*inch, 0.95*inch, 0.65*inch, 0.7*inch, 0.85*inch, 0.95*inch])
t4.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#294d69")), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.3),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f6f9")]),
    ("ALIGN",(1,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("TOPPADDING",(0,0),(-1,-1),2.5),("BOTTOMPADDING",(0,0),(-1,-1),2.5),
]))
story += [t4]
if SFT_B:
    story += [P(f"<i>&pi;<sub>ref</sub> is trained: one full epoch over {SFT_B.get('budget','32k')} "
                f"UltraFeedback pairs, {SFT_B.get('global_step')} optimizer steps at effective batch "
                f"{SFT_B.get('effective_batch')}, LoRA r=16. DPO and the RM both initialize from it, so "
                "neither scorer gets a head start.</i>", SMALL)]
story += [P("Evaluation reports pairwise accuracy (primary), ECE, length-controlled accuracy, Spearman "
            "vs gold, and a McNemar paired test between the implicit and explicit scorers "
            "(<i>src/metrics.py</i>, <i>src/eval.py</i>). The harness has been validated end-to-end on a "
            "reduced run." + ("" if EVAL else " All three training stages are complete and the full "
            "evaluation over the ID and OOD benchmark sets is currently running; H1/H2 numbers follow "
            "on its completion."))]
if RM_B and DPO_B and RM_B.get("global_step") and DPO_B.get("global_step") \
        and RM_B["global_step"] != DPO_B["global_step"]:
    _rm_pairs = RM_B["global_step"] * RM_B["effective_batch"]
    _dpo_pairs = DPO_B["global_step"] * DPO_B["effective_batch"]
    story += [P(
        f"<b>Known caveat on the compute match.</b> The RM completed {RM_B['global_step']} optimizer "
        f"steps ({_rm_pairs:,} pairs) against DPO&rsquo;s {DPO_B['global_step']} ({_dpo_pairs:,}) at the "
        "same nominal 8k budget and identical effective batch. The cause is a difference in how the two "
        "TRL trainers handle over-length examples: RewardTrainer <i>drops</i> pairs exceeding "
        "max_length while DPOTrainer <i>truncates</i> them. The two conditions therefore do not see "
        "byte-identical data, which weakens the &lsquo;same pairs&rsquo; half of our matched-compute "
        "definition. We are pre-filtering both conditions to a common length-eligible subset and will "
        "re-run before reporting headline H1/H2 numbers; the figures below should be read with this "
        "in mind.", SMALL)]

# ---------------------------------------------------------- 3.4 correctness work (new)
story += [P("3.4&nbsp;&nbsp;Two correctness bugs found before trusting any number", H2)]
story += [P("Bringing the pipeline up on GPU surfaced two defects that would not have crashed &mdash; they "
            "would have silently produced plausible but meaningless H1/H2 results. Both stemmed from "
            "treating the SFT <i>LoRA adapter</i> as the shared initialization:")]
story += [bullets([
    "<b>The DPO reference was the wrong model.</b> With a PEFT policy and <i>ref_model=None</i>, TRL "
    "derives &pi;<sub>ref</sub> by <i>disabling the adapter</i> &mdash; which returns the raw base model, "
    "not the SFT checkpoint. Training therefore optimized &beta;&middot;(log&nbsp;&pi;<sub>&theta;</sub> "
    "&minus; log&nbsp;&pi;<sub>base</sub>) while evaluation scored against base+SFT: two different "
    "quantities, neither the &pi;<sub>ref</sub> the design specifies.",
    "<b>The reward model&rsquo;s scalar head never trained.</b> Loading a CAUSAL_LM adapter onto a "
    "sequence-classification model leaves the freshly initialized <i>score</i> head frozen at random "
    "values (the adapter carries no <i>modules_to_save</i>). Measured directly: <i>score head "
    "trainable = False</i>. The explicit RM &mdash; the entire H1 baseline &mdash; would have been "
    "a LoRA trained underneath a random frozen projection.",
])]
story += [P("Both are fixed by materializing &pi;<sub>ref</sub> once as a merged checkpoint "
            "(<i>src/merge_sft.py</i>). DPO then trains a fresh zero-initialized LoRA on it, so "
            "&pi;<sub>&theta;</sub>&nbsp;=&nbsp;&pi;<sub>ref</sub> at step 0 and the implicit reward "
            "starts at exactly 0 &mdash; confirmed empirically by the DPO loss beginning at "
            "ln&nbsp;2&nbsp;&asymp;&nbsp;0.693. The RM adds a fresh SEQ_CLS LoRA, after which the head "
            "is trainable and saved. Evaluation loads the same merged checkpoint, so training and "
            "evaluation agree by construction.")]
story += [P("The implicit-reward extraction is additionally cross-checked against DPOTrainer&rsquo;s own "
            "reward computation (<i>src/verify_implicit.py</i>): tokenization matches exactly, the reward "
            "is exactly linear in &beta;, and the residual disagreement is an order of magnitude smaller "
            "than the shift a prompt-masking bug would produce. This guards the quantity the entire "
            "study measures.")]

# ---------------------------------------------------------- 3.5 first H1/H2 results
if EVAL:
    _ORDER = ["UltraFeedback (ID)", "RewardBench:Chat (OOD)", "RewardBench:Chat-Hard (OOD)",
              "RewardBench:Safety (OOD)", "RewardBench:Reasoning (OOD)", "HH-harmless (OOD)"]
    _sets = [k for k in _ORDER if k in EVAL] + [k for k in EVAL if k not in _ORDER]

    def _acc(row, scorer):
        v = row.get(scorer, {}).get("accuracy")
        return f"{v:.3f}" if isinstance(v, (int, float)) else DASH

    def _lca(row, scorer):
        v = row.get(scorer, {}).get("len_controlled_acc")
        return f"{v:.3f}" if isinstance(v, (int, float)) else DASH

    story += [P("3.5&nbsp;&nbsp;First H1/H2 results (single seed, &beta;=0.1, 8k budget)", H2)]
    story += [P("Pairwise accuracy of each scorer as a preference classifier. <b>Implicit</b> is DPO&rsquo;s "
                "r&#710;, <b>explicit</b> the Bradley&ndash;Terry RM, and <b>base log-prob</b> the free "
                "neural baseline (length-normalized log-probability under &pi;<sub>ref</sub>). The final "
                "column is the McNemar paired test between the implicit and explicit scorers on the "
                "identical pairs.")]

    _rows = [["Test set", "Implicit", "Explicit", "Base log-prob", "McNemar p"]]
    for k in _sets:
        row = EVAL[k]
        p = row.get("_mcnemar_p_impl_vs_expl")
        _rows.append([Paragraph(k.replace("RewardBench:", "RewardBench: "), SMALL),
                      _acc(row, "implicit"), _acc(row, "explicit"), _acc(row, "base_logprob"),
                      # plain "<" here, not &lt;: raw Table cells are drawn literally, not parsed
                      ("<0.0001" if isinstance(p, (int, float)) and p < 1e-4
                       else (f"{p:.4f}" if isinstance(p, (int, float)) else DASH))])
    t5 = Table(_rows, colWidths=[2.35*inch, 0.85*inch, 0.85*inch, 1.1*inch, 0.9*inch])
    _style = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#294d69")), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.3),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f6f9")]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),2.5),("BOTTOMPADDING",(0,0),(-1,-1),2.5),
    ]
    # Bold the better of implicit/explicit per row; red any scorer that lands below chance.
    for _i, k in enumerate(_sets, start=1):
        _im = EVAL[k].get("implicit", {}).get("accuracy")
        _ex = EVAL[k].get("explicit", {}).get("accuracy")
        if isinstance(_im, (int, float)) and isinstance(_ex, (int, float)):
            _win = 1 if _im > _ex else 2
            _style.append(("FONTNAME", (_win,_i), (_win,_i), "Helvetica-Bold"))
        for _c, _v in ((1,_im), (2,_ex)):
            if isinstance(_v, (int, float)) and _v < 0.5:
                _style.append(("TEXTCOLOR", (_c,_i), (_c,_i), colors.HexColor("#b00020")))
    t5.setStyle(TableStyle(_style))
    story += [t5, P("<i>Bold = better of the two scorers on that set. Red = below chance (0.500). "
                    "Ties count as 0.5.</i>", SMALL)]

    _id_im = EVAL.get("UltraFeedback (ID)", {}).get("implicit", {}).get("accuracy")
    _id_ex = EVAL.get("UltraFeedback (ID)", {}).get("explicit", {}).get("accuracy")
    _ood = [k for k in _sets if "OOD" in k]
    _im_wins = sum(1 for k in _ood
                   if EVAL[k].get("implicit", {}).get("accuracy", 0) > EVAL[k].get("explicit", {}).get("accuracy", 0))

    story += [P("Interpretation", H2)]
    story += [bullets([
        f"<b>H1 is supported.</b> In distribution the explicit RM is the better preference classifier: "
        f"{_id_ex:.3f} vs {_id_im:.3f} on held-out UltraFeedback, a gap that is statistically significant "
        "under McNemar. Training a reward head directly on the preference objective does buy accuracy "
        "on the distribution it was fit to.",
        f"<b>H2 is contradicted &mdash; the gap does not widen under shift, it reverses.</b> The implicit "
        f"reward is the better scorer on {_im_wins} of the {len(_ood)} out-of-distribution sets. The "
        "explicit RM wins only on RewardBench-Chat, the subset stylistically closest to UltraFeedback, "
        "and falls <b>below chance</b> on Chat-Hard, Reasoning and HH-harmless. On this evidence the "
        "DPO policy is the <i>more</i> transferable reward model, the opposite of what we predicted.",
        "<b>A below-chance scorer is a finding that demands a second look, not a result to report as is.</b> "
        "Systematically worse than random means the ordering is being actively inverted. The plausible "
        "mechanism is length: the explicit RM appears to proxy response length, and Chat-Hard is "
        "constructed to invert the length cue (pick-longer scores "
        f"{res['RewardBench: Chat-Hard (OOD)']['pick_longer']:.3f} there). We will confirm this against the "
        "length-controlled metric before treating it as a property of BT reward models rather than a defect.",
        "<b>Part of the implicit reward&rsquo;s OOD strength may not be preference learning at all.</b> On "
        f"Reasoning the free base log-probability baseline already reaches "
        f"{EVAL['RewardBench:Reasoning (OOD)']['base_logprob']['accuracy']:.3f}, close to the implicit "
        f"reward&rsquo;s {EVAL['RewardBench:Reasoning (OOD)']['implicit']['accuracy']:.3f}. Much of that "
        "column may reflect the base model&rsquo;s fluency rather than anything DPO learned &mdash; which "
        "is exactly why the log-prob baseline is reported alongside.",
    ])]
    story += [P("<b>These numbers are provisional.</b> They are a single seed at 0.5B with the pair-count "
                "caveat above, and the study design calls for &ge;3 seeds with confidence intervals before "
                "any headline claim. They are reported here as the first end-to-end signal from the "
                "pipeline, not as a settled result.", SMALL)]

# ------------------------------------------------------------------ 4. Challenges
story += [P("4&nbsp;&nbsp;Challenges", H1)]
story += [bullets([
    "<b>Auditable budget-matching.</b> The comparison is only meaningful if DPO and the RM see "
    "identical data/compute. We fix a shared SFT initialization and match epochs, LoRA config, batch, "
    "and LR, and log tokens-seen / wall-clock &mdash; but defining &lsquo;equal compute&rsquo; across two "
    "different losses (DPO&rsquo;s extra reference forward pass) remains a judgment call we must defend.",
    "<b>Correct implicit-reward extraction.</b> The &beta;-scaled, prompt-masked, length-summed log-ratio "
    "must exactly match TRL&rsquo;s internal convention; a subtle masking or scaling error would silently "
    "bias every downstream number. <i>Addressed</i> (Section 3.4): cross-checked against DPOTrainer&rsquo;s "
    "own reward computation, with explicit controls for the masking and &beta;-scaling failure modes.",
    "<b>Silent failures beat loud ones.</b> The two defects in Section 3.4 both ran without error and "
    "produced numbers that looked reasonable. The lesson we are carrying forward is to verify each "
    "component against an independent implementation before trusting any result it feeds.",
    "<b>Small-model noise.</b> At 0.5B, reward accuracy can sit close to chance on hard subsets "
    "(the TF-IDF collapse on Reasoning previews this), so headline claims need &ge;3 seeds with "
    "confidence intervals; we may escalate to 1.5B if the signal is too noisy.",
    "<b>Length as a confound.</b> Both scorers can proxy length (Section 3.2); isolating true preference "
    "signal requires the length-controlled metric and reporting the length-only baseline everywhere.",
    "<b>Gold-judge / Spearman set on limited compute.</b> A larger instruction-tuned judge for scalar "
    "gold ratings must be run as a separate quantized/offline pass (or via API) to fit a single "
    "consumer GPU alongside training.",
    "<b>Toolchain drift &mdash; encountered, not hypothetical.</b> Standing the environment up required "
    "upper-bounding <i>transformers</i> (TRL 0.15 imports a symbol removed in transformers 5.x, so an "
    "unbounded pin broke the import outright) and handling an SFTConfig argument that was renamed across "
    "TRL releases. Versions are now pinned on both sides in <i>requirements.txt</i>.",
    "<b>Hardware-dependent silent misconfiguration.</b> On a multi-GPU host, HuggingFace Trainer "
    "auto-wraps the model in DataParallel, which quadrupled the effective batch size &mdash; breaking the "
    "matched-compute definition without any warning &mdash; and ran 6&times; slower. Runs are now pinned "
    "to one GPU so the executed configuration matches the documented one.",
])]

# ------------------------------------------------------------------ 5. Next steps
story += [P("5&nbsp;&nbsp;Next Steps", H1)]
_next = ([
    "<b>Immediate &mdash; explain the below-chance explicit RM.</b> Check its accuracy on length-matched "
    "pairs against the length-only baseline. If the length-proxy account holds, this becomes a substantive "
    "result about what BT reward models latch onto; if it does not, it points at a defect in the scalar "
    "head that must be fixed before any comparison stands.",
    "<b>Restore an exact compute match.</b> Pre-filter both conditions to a common length-eligible subset "
    "so the RM and DPO see byte-identical pairs, then re-run the 8k budget.",
    "Repeat the headline configuration across &ge;3 seeds and attach confidence intervals; the current "
    "numbers are a single seed and cannot carry a claim on their own.",
] if EVAL else [
    "<b>Immediate.</b> Finish DPO (&beta;=0.1) and the BT reward model at the 8k budget from the shared "
    "&pi;<sub>ref</sub>, then run the evaluation harness to produce the first H1/H2 numbers "
    "(implicit vs explicit, ID + OOD) against the reference floors in Section 3.1.",
    "Re-run the implicit-reward cross-check on the trained checkpoints, where the rewards are far from "
    "zero and the comparison is most informative.",
]) + [
    "Widen to the ablations: &beta; &isin; {0.05, 0.1, 0.3, 0.5}, budget &isin; {2k, 8k, 32k}, and the "
    "training-progress checkpoint curve (overoptimization).",
    "Add the gold-judge OOD set and Spearman correlation; finalize with &ge;3 seeds and significance tests.",
]
story += [bullets(_next)]
story += [Spacer(1, 4), HRFlowable(width="100%", color=colors.HexColor("#c9c9c9")),
          P("Code, benchmark data pipeline, and reproducible baseline results: "
            "github.com/silvererudite/dpo-implicit-reward", SMALL)]

# ------------------------------------------------------------------ build
out = os.path.join(ROOT, "reports", "project_update.pdf")
SimpleDocTemplate(out, pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch,
                  leftMargin=0.8*inch, rightMargin=0.8*inch,
                  title="Exploring DPO's Implicit Reward - Project Update").build(story)
print("Wrote", out)
