# Week 6 Presentation: Surface-Aware Dental Caries Classification

> **Presenter Guide** — 5 slides, ~15 minutes  
> Hero shots: `week6/presentation_hero_shots/`

---

## Slide 1: The Problem — The 2D Challenge

### Bullet Points
- Periapical X-rays are **2D shadow projections** of 3D anatomy
- Standard detection models output a **bounding box** — "there is caries here"
- But a bounding box **cannot tell the dentist which surface** is affected
- **Mesial? Distal? Occlusal?** — the surface determines the treatment plan
- A Class II (proximal) restoration is a fundamentally different procedure from a Class I (occlusal)

### Speaker Script
> "Let me start with the fundamental limitation we're dealing with. A periapical radiograph is, at its core, a two-dimensional shadow of a three-dimensional object. When we run a standard YOLO detection model on these images, we get a bounding box — essentially, a rectangle that says *'caries is somewhere in here.'*
>
> But here's the problem: that rectangle tells us nothing about **where on the tooth** the decay actually sits. Is it on the mesial surface? The distal? The occlusal? This distinction isn't academic — it directly determines the treatment plan. A mesial-occlusal restoration requires a completely different preparation than a simple occlusal filling.
>
> **Location is everything in diagnosis.** And that's the gap I set out to close."

### Visual Cue
- Show a raw X-ray with a YOLO bounding box drawn around a carious tooth
- Annotate with a question mark: **"M? O? D?"**
- Any case from `evaluation_output/` showing the global view (top half of dashboard)

---

## Slide 2: The Solution — PCA + Multi-Zone Point-Cloud Voting

### Bullet Points
- **Step 1 — PCA Orientation Normalization**
  - Compute the Eigenvectors of each tooth's segmentation polygon
  - The 1st principal component = the tooth's "long axis"
  - Rotate all coordinates into a canonical frame → teeth always point the same way
- **Step 2 — M-C-D Zone Division**
  - Divide the PCA-aligned bounding box into 3 equal zones: Mesial | Central | Distal
  - Quadrant-aware flipping: FDI Q1/Q4 (right) vs Q2/Q3 (left) determines which end is Mesial
- **Step 3 — 5% Point-Cloud Voting** (replacing single-centroid)
  - Project **every caries pixel** into the rotated coordinate space
  - Count the fraction of points in each zone (M, C, D)
  - **Any zone ≥ 5%** is reported — following the G.V. Black "Any Involvement" standard
  - Connected-component noise removal (clusters < 15 px discarded)
- **Result**: `MOD`, `MO`, `DO`, `M`, `D`, `O` — clinically actionable labels

### Speaker Script
> "So how do we go from a bounding box to a surface-specific diagnosis? Three steps.
>
> **First, PCA.** Every tooth in an X-ray sits at a different angle — some tilted, some rotated. I use Principal Component Analysis on each tooth's segmentation polygon to find the eigenvectors — essentially, the tooth's natural long axis. I then rotate everything into a standardized frame. Now, every tooth 'points' the same way, regardless of its original orientation in the image.
>
> **Second, zone division.** Once the tooth is aligned, I split its PCA-rotated bounding box into three equal zones: Mesial, Central, and Distal — the M-C-D zones. Crucially, which end is 'Mesial' depends on the FDI quadrant. For teeth on the patient's right side — quadrants 1 and 4 — mesial is anatomically to the left in the rotated frame. For the left side, it flips. This quadrant-aware logic was actually one of the key fixes that took our surface accuracy from 27% to 87%.
>
> **Third — and this is the real breakthrough — point-cloud voting.** The old approach used a single centroid: one point, one zone, one label. That completely fails for large cavities that span multiple surfaces. Instead, I now project *every* caries pixel into the rotated space and count how many land in each zone. If 5% or more of the points touch a zone, that zone is reported. This 5% threshold follows the G.V. Black classification standard — *any involvement* of a surface counts. I also apply connected-component filtering to remove noise — isolated clusters smaller than 15 pixels are discarded before voting.
>
> The output is a clinically meaningful label: M, D, O, MO, DO, or MOD."

### Visual Cue
- Show the **bottom half** of a dashboard image (PCA panel) from a Complex Win case
- Ideal: a **MOD case** like `2_The_Complex_Win/case_57_MOD_...png` or `case_365_MOD_...png`
- Point out: blue PCA axis, red zone dividers, cyan/yellow/magenta caries points in M/C/D zones

---

## Slide 3: The Evidence — Hero Shots

### Bullet Points
- **The Perfect Match** — TP, correct surface, high IoU
  - Model and ground truth agree completely
  - Establishes **baseline trust** in the system
- **The Complex Win** — Multi-surface TP (MO / DO / MOD)
  - Old centroid logic → would label as "O" (center only)
  - New point-cloud voting → correctly identifies **extension into M or D**
  - 328 multi-surface cases detected across 500 patients
- **Why it matters**: Multi-surface caries changes the restoration class
  - Class I (O only) → Class II (MO/DO/MOD) = more invasive preparation

### Speaker Script
> "Let me show you the evidence.
>
> First, a Perfect Match. *[show image]* Here, the model detected the caries, assigned it to the correct tooth, and — critically — identified the correct surface. The ground truth says Distal-Occlusal; the model says Distal-Occlusal. The IoU with the tooth polygon is 0.26, and the surface labels align perfectly. This is our trust-builder.
>
> But the real story is here — the Complex Wins. *[show MOD case]* Look at this case. The caries lesion spans from the mesial marginal ridge, across the occlusal surface, all the way to the distal. If I had used the old centroid-based approach — which just drops a single point at the center of mass — it would have landed squarely in the Central zone and reported 'Occlusal.' That's technically not wrong, but it's dangerously incomplete. It misses the proximal extensions entirely.
>
> With point-cloud voting, the system projects all the caries pixels into the PCA-aligned space. You can see the cyan points in the Mesial zone, yellow in the Central, and magenta in the Distal. The fractions come out to something like M=22%, C=45%, D=33% — all well above our 5% threshold. The system correctly reports **MOD**.
>
> This matters clinically: an Occlusal-only lesion is a Class I restoration. An MOD lesion is a Class II — a fundamentally more invasive preparation. Getting this distinction right has direct implications for treatment planning.
>
> Across all 500 cases, the model identified **328 unique cases** with multi-surface caries — 151 MOD, 131 MO, and 46 DO."

### Visual Cue
- **Left**: A Perfect Match dashboard from `1_The_Perfect_Match/` (e.g., `case_217_DO_iou0.26.png`)
- **Right**: A Complex Win MOD case from `2_The_Complex_Win/` (e.g., `case_57_MOD_...png`)
- Zoom into the PCA panel to show the three-color caries distribution across zones
- Optionally: side-by-side "centroid would say O" vs "voting says MOD" annotation

---

## Slide 4: The Performance

### Bullet Points
- **Detection Performance** (500 cases, 2272 annotations)
  - True Positives: 1,821
  - False Positives: 293
  - False Negatives: 158
  - **Precision: 0.86** — when the model flags caries, it's right 86% of the time
  - **Recall: 0.92** — the model catches 92% of all actual caries
  - **F1-Score: 0.89** — strong balance between precision and recall
- **Surface Classification Accuracy: 87.31%**
  - Week 5 baseline (centroid-only): **27%**
  - Week 6 (PCA + point-cloud voting): **87%** — a 3.2× improvement
- **Transparency**: Full confusion matrix and per-case CSV available

### Speaker Script
> "Now, the numbers.
>
> Across 500 cases and 2,272 ground-truth annotations, the system achieved an F1-Score of **0.89** — with precision at 0.86 and recall at 0.92. That recall number is particularly important in a screening context: we're catching 92% of all caries in the dataset.
>
> But the headline metric for this week is **Surface Accuracy**. In Week 5, using a naive centroid-based approach, surface accuracy was just 27%. With the PCA normalization, quadrant-aware orientation, and 5% point-cloud voting threshold, that number jumped to **87.31%**. That's a 3.2-times improvement — and it's the difference between a system that can only say 'caries exists' and one that can say 'caries exists *on the mesial-occlusal surface*.'
>
> For full transparency, every case has its own evaluation CSV and validation dashboard. The confusion matrices are published alongside the results. Nothing is hidden."

### Visual Cue
- A **metrics summary table** (can be a simple styled slide)
- Optional: bar chart showing 27% → 87% surface accuracy progression
- Reference: `evaluation_output/evaluation_summary.json` for exact numbers

---

## Slide 5: Discussion — The "AI Eye"

### Bullet Points
- **293 False Positives** — but not all are truly "wrong"
- **4 high-confidence FP cases** (confidence > 0.85):
  - Case 249 (conf=0.96), Case 72 (conf=0.94), Case 365 (conf=0.85), Case 90 (conf=0.85)
- Hypothesis: These may be **incipient (early-stage) caries** that were not annotated in the ground truth
  - Radiographic subtlety: early enamel demineralization is easily missed by human readers
  - Inter-examiner variability in caries detection is well-documented (κ = 0.50–0.70 in literature)
- **Reframing**: The model as a **"Second Opinion Screening System"**
  - Not replacing the dentist — **augmenting** clinical attention
  - Flagging regions for closer inspection, especially in high-volume screening
- **Remaining errors** (low-confidence FP + FN):
  - Often involve fillings misclassified as caries (radiopaque overlap)
  - Or small periapical lesions outside the model's training distribution

### Speaker Script
> "Finally, let's talk about the errors — because I think they tell the most interesting story.
>
> We have 293 false positives. At first glance, that sounds like a lot. But let me zoom in on a specific subset: 4 cases where the model flagged caries with **very high confidence** — above 0.85 — but the ground truth said 'no caries there.'
>
> *[show Case 249, conf=0.96]* Look at this case. The model is 96% confident there is proximal caries here. The human annotator didn't mark it. Now, one interpretation is that the model is simply wrong. But another — and I think more clinically interesting — interpretation is that the model may be detecting **incipient caries**: early-stage enamel demineralization that is radiographically subtle and easily missed on first read.
>
> This isn't speculation — the literature on inter-examiner agreement in caries detection consistently reports kappa values between 0.50 and 0.70. Dentists disagree with each other on borderline cases all the time.
>
> So I want to reframe these high-confidence false positives. Rather than treating them as failures, I see them as evidence that this system could serve as a **second-opinion screening tool** — not replacing clinical judgment, but augmenting it. Flagging regions that deserve a second look, especially in high-volume settings like public health screenings or insurance reviews.
>
> As for the remaining errors — the low-confidence false positives and the false negatives — these tend to involve metallic restorations being misread as caries, or very small lesions at the limit of radiographic resolution. These are areas for continued model refinement.
>
> Thank you. I'm happy to take questions."

### Visual Cue
- Show all 4 AI Eye cases from `3_The_AI_Eye_Potential/`:
  - `case_249_Proximal_conf0.96.png`
  - `case_72_Occlusal_conf0.94.png`
  - `case_365_Occlusal_conf0.85.png`
  - `case_90_Proximal_conf0.85.png`
- Zoom into the suspicious region on the X-ray; circle the area the model flagged
- Optional: An Honest Mistake case from `4_The_Honest_Mistake/` for contrast (e.g., a filling misread)

---

## Quick Reference — File Locations

| Asset | Path |
|---|---|
| Hero shots | `week6/presentation_hero_shots/` |
| Perfect Match (20) | `presentation_hero_shots/1_The_Perfect_Match/` |
| Complex Win (328) | `presentation_hero_shots/2_The_Complex_Win/` |
| AI Eye (4) | `presentation_hero_shots/3_The_AI_Eye_Potential/` |
| Honest Mistake (20) | `presentation_hero_shots/4_The_Honest_Mistake/` |
| Full dashboards | `week6/evaluation_output/case {N}/` |
| Evaluation CSV | `week6/evaluation_output/evaluation_results.csv` |
| Summary JSON | `week6/evaluation_output/evaluation_summary.json` |
| Manifest | `presentation_hero_shots/hero_shots_manifest.json` |
