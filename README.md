**VideoFrameReconstruction**

1. Install dependencies:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

requirements.txt
opencv-python-headless
numpy
scikit-image
tqdm

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


We reconstruct the original frame order by computing directed pairwise likelihoods between frames using a combination of global color histogram similarity, ORB feature-match strength, and SSIM between the right half of a frame and the left half of another (to detect spatial continuity). These complementary cues produce a directed weighted graph where a high weight from frame i to frame j indicates j probably follows i. We then greedily assemble frames into a single chain by iteratively merging chains using the highest-weight edges while avoiding cycles. Pairwise scoring is parallelized across cores; the assembly is linear-time after scoring. The method is robust across panning, object movement, and illumination changes and can be improved via CNN embeddings, optical-flow checks, or local refinement passes.
