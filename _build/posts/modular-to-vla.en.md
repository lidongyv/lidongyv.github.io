I group the work since 2021 by one question: is perception inside the policy? That gives three generations — modular perception, two-stage end-to-end, one-stage VLA. The technical details differ a lot, but every generation hit the same wall: **the trade-off between single-model and multi-task metrics**.

## Generation one: modular — good metrics, hard to learn from

At NIO it was textbook modular work: Mono3D detection and tracking plus lane detection plus post-processing, with a data loop and an evaluation harness pushing each module's numbers up separately. For association post-processing we used rule-based association over 7 cameras and 1 LiDAR, then replaced the rules and bipartite matching with a learnable cost metric — at a 20% 3-D error rate we reached 0.03 ID switches and 86% recall.

The advantage is that every metric can be explained and rolled back. The cost is equally clear: **errors accumulate**, and the interfaces between modules are defined by hand, so the model cannot learn anything beyond them.

## Generation two: two-stage end-to-end, perception becomes tokens

At NVIDIA the core move was reorganising perception so it fits E2E: LiDAR early fusion, detection, MOT upgraded to object prediction, and a future tokenizer that supports pre- and post-training. Lane detection took the long road: segment-based (what-you-see-is-what-you-get, pushed with super-resolution, BEV multi-scale fusion and dynamic multi-resolution DCN) → query-based (navi info, intersections, split/merge, lane changes) → token-based (all perception elements tokenized autoregressively).

The token-based step is the watershed. It is what let two-stage become one-stage, because **once perception elements are tokens, planning can read them like language**.

## Generation three: one-stage, planning reads the image stream

In one-stage, planning moved from consuming perception outputs to consuming the image stream, the perception tokenizer was removed and demoted to a debugger task. Trajectory prediction changed shape the whole way: single-mode → multi-mode regression → autoregressive → diffusion, then RL post-training and closed-loop training against a world model.

## The trap that kept coming back

Across three generations the same question appeared three times: **how do you put several tasks in one model without having them drag each other down?**

At Neolix the answer was a group query-based model with single-head multi-task training: multi-head to single-head, so data balancing becomes something the model can configure; and the balancing happens at token level (BEV tokens / image tokens) rather than at task-loss level. Query grouping had already been validated at Horizon — it closed the single-task / multi-task metric gap, and replacing the two-stage detector with a query-based one for discrete elements (arrows, crosswalks, junctions) held the score while cutting compute by more than 30%.

My conclusions:

1. **The crux of multi-task is the representation layer, not loss weighting.** Task-loss weighting patches the result; token-level rebalancing fixes the cause.
2. **A single task is a subset of multi-task.** If an architecture cannot run a single task as multi-task with N=1, it will eventually split into two codebases.
3. **Being able to add data at any time matters more than training once to the best number.** Frozen-head pretraining of the backbone, frozen-backbone fine-tuning of the perception head — in engineering terms this controllable iteration is worth far more than a leaderboard gain.

Next I will write about the data loop, because in VLA the model architecture is actually the easiest part to catch up on.
