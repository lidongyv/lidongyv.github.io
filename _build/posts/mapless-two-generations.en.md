"Mapless" is easily misread as "delete the map module". But once the map is gone, everything the map used to provide for free — topology, junction structure, lane ownership — has to be learned by the model.

## First generation: segment-based, what you see is what you get

The first approach was what-you-see-is-what-you-get: the model emits lane lines and road edges only, no centerlines, and we pushed geometric accuracy with super-resolution, BEV multi-scale fusion and dynamic multi-resolution DCN. To get GT that lined up, we built an HD-map → local-BEV GT pipeline (2M images, 12M lane instances), lifting segment-based lane detection from AP 13% to 60% and EPE from 1.5 m to 0.18 m.

The ceiling of this route is obvious: **it can only see what it sees**. Where a junction is occluded, where the far field is barely visible, where the lane marking is broken, the model has no way to fill in.

## Second generation: query-based, from seeing to inferring

The query-based approach represents a lane as a multi-level regression over lane → lane line → point chain. Post-processing goes away, effort moves upstream, compute stays flat while F1 improves 15%. More importantly it can take navi info: go straight or turn left at the junction, split or merge, change lane — those upstream decisions become model inputs, and the model starts to *guess* the parts it cannot see.

The price is that the data rules have to be rewritten: query-based is far more sensitive to labelling rules, data balancing and case-level fine-tuning. We built full-element, multi-task, same-source temporal datasets for road structure, and moved issue turnaround from D+10 to D+3.

## Third generation: token-based, preparing for end-to-end

The third generation was not about better numbers; it was about **getting lanes into the end-to-end stack**. Once perception elements are tokenized autoregressively, lane detection goes from "emit a BEV map" to "emit a token sequence", and can therefore share a decoder with trajectory prediction.

That step also dealt with the long tail: we ran three topology routes in parallel — autoregressive keypoint generation with a language model, MaskFormer-based topology keypoint segmentation, and a "reverse transformer" that turns query-based point-chain generation back into a segment algorithm. All three point at the same thing: **topology has to be generated explicitly, not post-processed out of a segmentation map**.

## What I take away

If only three lessons survive:

1. **Mapless is a topology problem, not a detection-accuracy problem.** Accuracy without topology cannot support a decision.
2. **Change the data rules before changing the representation.** Half of the query-based gain came from rewriting labelling rules and data balancing.
3. **The criterion for a generation change is "can it enter end-to-end"**, not "how much did the metric move". Token-based was worth doing because it made the next two generations possible.

After that comes the world-model line, which I write about elsewhere.
