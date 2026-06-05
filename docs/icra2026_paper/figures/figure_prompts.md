# Figure Drawing Prompts for ICRA 2026 Paper

## Figure 1: Overall System Framework

**Prompt for GPT Image / DALL-E:**

```
Create a clean, professional technical diagram for an academic paper (IEEE two-column format, white background, no decorative elements). The diagram shows a two-phase UAV interception system:

LEFT SIDE (Offline Phase, labeled "Offline"):
- A 2D map grid with several gray polygonal obstacles scattered across it
- Red dots marking "Voronoi Bottleneck Points" at geometric chokepoints between obstacles
- Blue dots marking "Coverage Points" distributed across free space
- An arrow labeled "Dubins A*" pointing to a time-field heatmap showing isochrone contours around each waypoint
- Box labeled "IsoMap Pre-computation" containing the time field

RIGHT SIDE (Online Phase, labeled "Online"):
- A flowchart going downward:
  1. Box: "Current State" (pursuer and evader positions)
  2. Arrow down to: "Candidate Construction" box (shows path splicing diagram)
  3. Arrow down to: "Transformer MARL Selection" box (shows attention mechanism selecting one of K candidates)
  4. Arrow down to: "Execute & Monitor" box
  5. A feedback loop arrow going back up, labeled: "DWA Deviation > Threshold → Replan"

BOTTOM: A dashed box connecting the Offline and Online phases, labeled "Hungarian Matching"

Style: Academic paper figure, black text on white background, arrows with clear labels, minimal color (use blue for pursuers, red for evaders, gray for obstacles). Font should be clean sans-serif.
```

**Alternative prompt (simpler):**

```
Draw a technical flowchart diagram for a robotics paper. White background, academic style.

The diagram has two main sections side by side:

LEFT: "Offline Phase"
- Shows a map with obstacles (gray polygons) and scattered waypoint dots
- A box labeled "Voronoi + Dubins A* → IsoMap"
- Arrow connecting map to IsoMap

RIGHT: "Online Phase" (flowchart)
- "State Input" → "Candidate Construction" → "MARL Selection" → "Execute"
- A loop arrow from "Execute" back to "Candidate Construction" labeled "DWA Replan"

Use blue circles for pursuers, red triangles for evaders, gray rectangles for obstacles.
Clean academic style, black and white with minimal blue/red accents.
```

---

## Figure 2: CQN Network Architecture

**Prompt:**

```
Create a neural network architecture diagram for an academic paper (IEEE format, white background, clean style).

The diagram shows the Concatenative Query Network (CQN) for multi-agent UAV interception:

TOP ROW (Input): 8 input branches shown as colored rectangles:
- "Self UAV" (green, 3-dim)
- "Allies" (light green, 3-dim × P-1)
- "Self Candidates" (blue, 8-dim × K)
- "Ally Candidates" (light blue, 8-dim × (P-1)×K)
- "Enemy (Self)" (red, 3-dim)
- "Enemy (Ally)" (pink, 3-dim × P-1)
- "Asset (Self)" (orange, 2-dim)
- "Asset (Ally)" (yellow, 2-dim × P-1)

MIDDLE ROW: Each input goes through an independent MLP → LayerNorm, producing 8 embedding vectors (all same hidden dim d=128)

CENTER: A large box labeled "Heterogeneous Multi-Head Attention (h=4)" with arrows from all 8 embeddings going into it. Inside, show cross-attention connections between the branches.

BOTTOM LEFT: "ConcatMLPFusion8" box → "Environment Representation h_env" (d-dim vector)

BOTTOM RIGHT: Two parallel outputs:
1. "Pointer Actor": h_env → dot product with candidate embeddings → softmax → action probabilities (masked)
2. "Centralized Critic": h_env + h_global (13-dim) → V_i value estimate

Style: Clean academic diagram, boxes with rounded corners, arrows showing data flow, labels in sans-serif font. Use consistent colors for the 8 input branches.
```

**Alternative (simpler):**

```
Draw a neural network architecture diagram for a paper. Shows:

Input: 8 colored input boxes in a row (Self, Allies, Candidates, Enemies, Assets)
Middle: Each input → MLP embedding → all go into a central "Multi-Head Cross-Attention" box
Output: Two branches - "Actor" (action selection) and "Critic" (value estimate)

White background, clean academic style, arrows showing flow direction.
```

---

## Figure 3: Scalability Plot

**Prompt:**

```
Create a line chart for an academic paper showing system scalability. Two subplots side by side:

LEFT SUBTITLE: "Capture Rate vs Agent Count"
- X-axis: "Number of Agents" with labels: "3v5", "5v8", "8v12", "12v20"
- Y-axis: "Capture Rate (%)" range 0-100
- 5 lines (different methods):
  - "Ours (MARL)" — solid blue line, highest values, slight decline with scale
  - "Cost-based" — dashed orange line, medium values, steeper decline
  - "Greedy" — dotted green line, lower values
  - "Random" — dash-dot red line, lowest values
  - "Vanilla MAPPO" — solid purple line, below Ours but above Greedy
- Add small markers (circles, squares, etc.) at data points
- Legend in upper right or lower left

RIGHT SUBTITLE: "Interception Time vs Agent Count"
- X-axis: same as left
- Y-axis: "Interception Time (steps)" range 0-400
- Same 5 lines, but "Ours (MARL)" has the lowest time (best)
- All lines increase with agent count

Style: Academic paper figure, white background, black axes, Times New Roman font, grid lines (light gray), no decorative elements. Use colorblind-friendly palette.
```

---

## Figure 4: Trajectory Visualization

**Prompt:**

```
Create a 2D trajectory visualization for a robotics paper. White background with light gray grid.

SCENE: A 2000×2000 map with:
- 5 gray polygonal obstacles of varying sizes scattered across the center
- 3 green stars at the bottom labeled "Facility 1/2/3" (protected assets)
- 3 blue circles at the top labeled "P1, P2, P3" (pursuer starting positions)
- 5 red triangles at the right labeled "E1, E2, E3, E4, E5" (evader starting positions)

TRAJECTORIES:
- Blue solid lines: Pursuer paths showing "gradual approach" — not directly chasing evaders, but moving toward intercept points (marked as small blue dots along the paths)
- Red dashed lines: Evader paths toward facilities
- One evader path shows a sudden turn (fake attack), marked with a yellow warning symbol
- At the fake attack point, show a blue arrow indicating the pursuer adapted its intercept point

INTERCEPT POINTS: Small blue dots along pursuer paths show where they changed direction to select new intercept points

ANNOTATIONS:
- Label "IsoMap Waypoint" near some blue dots
- Label "Fake Attack" near the sudden evader turn
- Label "Replan" near the pursuer's direction change
- Add a small legend in the corner

Style: Technical diagram, clean lines, academic paper quality. Use distinct colors for pursuers (blue) and evaders (red).
```

**Alternative (behavior comparison):**

```
Draw two side-by-side 2D trajectory plots for a paper:

LEFT: "Direct Pursuit (Fails)"
- Shows pursuers (blue) flying straight toward evaders (red)
- Evader suddenly turns, pursuer overshoots
- Dotted line shows missed interception
- Label: "Interception → Pursuit"

RIGHT: "Our Method (Succeeds)"
- Shows pursuers moving toward intermediate waypoints (small circles)
- Gradually approaching evaders from multiple angles
- Successful interception marked with a star
- Label: "Gradual Approach → Interception"

Both plots have obstacles (gray polygons) and facilities (green squares).
Clean academic style, white background.
```

---

## Figure 5: DWA Replanning Process

**Prompt:**

```
Create a sequence diagram showing the DWA-based replanning process for a UAV paper. 4 panels arranged horizontally:

PANEL 1: "Initial Plan"
- Shows pursuer (blue circle) and evader (red triangle)
- Dashed line: predicted evader path (from trajectory database)
- Solid blue line: pursuer's planned intercept path
- Label: "t = t_0"

PANEL 2: "DWA Monitoring"
- Same scene, but evader has started deviating
- Show DWA simulation: multiple thin gray lines emanating from evader position (simulated trajectories with different (v,ω))
- Red highlighted line: best-matching DWA trajectory
- Dashed line: original predicted path
- Arrow between them labeled "Deviation"
- Label: "t = t_1"

PANEL 3: "Deviation Detected"
- Show deviation measurement: distance between DWA prediction and actual path > threshold
- Red warning symbol
- Label: "deviation > 50 → Replan triggered"
- Label: "t = t_2"

PANEL 4: "New Plan"
- New candidate intercept points generated (blue dots)
- Pursuer selects new intercept point
- New blue path toward updated intercept point
- Hungarian matching updated
- Label: "t = t_3"

Style: Sequential diagram, white background, consistent blue/red/gray color scheme. Each panel clearly labeled with time step.
```

**Alternative (single frame):**

```
Draw a single technical diagram showing the DWA replanning concept:

- Center: Pursuer (blue) and Evader (red)
- From evader: multiple thin gray curves showing DWA simulated trajectories
- One red highlighted trajectory showing "best match"
- A dashed line showing "predicted path"
- Arrow between them showing "deviation > threshold"
- Below: Two states shown:
  1. "Before Replan": pursuer heading to old intercept point
  2. "After Replan": pursuer heading to new intercept point
- Label: "Trajectory Deviation Detection → Dynamic Replanning"

Clean academic style, white background, suitable for IEEE paper figure.
```

---

## General Tips for Image Generation

1. **Resolution**: Generate at 300 DPI or higher for print quality
2. **Size**: Each figure should fit in a single column (3.5 inches wide) or span full page (7 inches wide)
3. **Font**: Use sans-serif fonts (Arial, Helvetica) for labels
4. **Colors**: Use colorblind-friendly palette (blue, orange, green, red, purple)
5. **Format**: Save as PDF or PNG (high resolution)
6. **After generation**: Convert to PDF and place in `figures/` directory, then reference in LaTeX as:
   ```latex
   \begin{figure}[t]
       \centering
       \includegraphics[width=\columnwidth]{figures/framework.pdf}
       \caption{Overall framework of the proposed method.}
       \label{fig:framework}
   \end{figure}
   ```

## Recommended Tools

- **GPT-4 with DALL-E**: Good for technical diagrams, can iterate
- **Midjourney**: Better for artistic/visual figures
- **draw.io / diagrams.net**: Best for flowcharts and architecture diagrams (free, exports to PDF)
- **TikZ (LaTeX)**: Best for publication-quality figures, but steep learning curve
- **matplotlib**: Best for plots and charts (already in your Python environment)
- **Inkscape**: Good for vector editing and cleanup
