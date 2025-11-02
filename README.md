# Tecdia_Pooja_VIT
**VideoFrameReconstruction**

1. Install dependencies:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# requirements.txt
# opencv-python-headless
# numpy
# scikit-image
# tqdm


2. Run:

python reconstruct_frames.py --input jumbled_video.mp4 --out reconstructed.mp4 --workdir work


3. Outputs in work/:

frames/ — extracted frames

reconstructed_order.txt — final order of frame indices

execution_log.txt — total runtime and parameters

reconstructed.mp4 — reconstructed video

-------------------------------------------------------------------------

Goal: reconstruct the original temporal order of 300 shuffled frames.

Key idea:

Build a directed pairwise similarity score S(i → j) that expresses how likely frame j follows frame i.

Use multiple complementary cues (global color similarity, local structural similarity across the right/left border, and ORB feature-match strength). Combining cues helps with different video content (camera pan, object motion, lighting).

Assemble frames into a full chain using a greedy graph-assembly algorithm that links best successors while avoiding cycles. This is efficient and works well in practice for single-shot videos. For 300 frames this is tractable; pairwise scoring is the main cost and is easily parallelizable.

Why these cues?

Color histograms are cheap and robust for scenes that don’t change abruptly.

Right-half-of-frame → left-half-of-next-frame SSIM detects continuity for lateral motion or camera pans.

ORB feature matches capture structural consistency where features persist across adjacent frames.

Combining them reduces failure modes of any single cue.

Complexity:

Pairwise computation naive O(n²) comparisons. For n=300 this is 90k pairs — acceptable on modern hardware and simple to parallelize.

Assembly via greedy linking runs ~O(n log n) after pairwise stage.

Extract frames from jumbled_video.mp4 → frames/000000.png, ...

Resize frames to a working size (e.g., width 640) for speed but preserve aspect ratio.

For each frame compute:

HSV histogram (concatenate H and S hist bins).

ORB keypoints & descriptors.

Optionally store grayscale and left/right halves for SSIM.

For each ordered pair (i, j) compute directed similarity S(i→j):

hist_sim = Bhattacharyya / correlation between HSV histograms.

orb_matches_ratio = (# good ORB matches) / min(k1, k2) where “good” uses Lowe ratio test.

half_ssim = SSIM between right-half of frame i and left-half of frame j (captures spatial continuity).

Combine: S = w1 * hist_sim + w2 * orb_matches_ratio + w3 * half_ssim. (weights configurable.)

For each frame i, choose top-K candidate successors by S (K small like 5 or 10).

Greedy assembly:

Initialize each frame as chain of length 1.

Sort all candidate edges by decreasing S.

Iteratively take best edge (u → v): if u is currently chain-end (right end) and v is currently chain-start (left end) and linking would not create a cycle, merge the two chains by appending v's chain after u's chain.

Continue until one chain of length n emerges (or no mergeable edges remain).

Output the reconstructed sequence and re-encode frames to reconstructed.mp4 at original FPS.

This linking approach is essentially greedy maximum-weight chain assembly and works well for single-shot videos.
