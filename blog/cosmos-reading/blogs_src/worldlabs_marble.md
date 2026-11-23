# Marble: A Multimodal World Model (World Labs, Fei-Fei Li)

Source: World Labs blog — https://www.worldlabs.ai/blog/marble-world-model
Date: November 12, 2025

Spatial intelligence is the next frontier in AI, demanding powerful world models. World models should reconstruct, generate, and simulate 3D worlds; and allow both humans and agents to interact with them.

Marble is a first-in-class generative multimodal world model, now generally available. Highlights:
- Multimodal input: Marble creates 3D worlds from TEXT, IMAGES, VIDEO, or COARSE 3D LAYOUTS; and lets you interactively edit, expand, and combine worlds.
- Export: once generated, 3D worlds can be exported as GAUSSIAN SPLATS, MESHES, or VIDEOS.
- Marble Labs: creative hub with case studies, tutorials, documentation.

## The Marble World Model
Human experience of the world is inherently multimodal (sight, sound, touch, language build a mental model). World models should work similarly: massively multimodal, able to lift whatever input signals are available into a full 3D world, and iteratively update understanding as new information arrives.

## Text and Image to World
Marble can create a full 3D world from a single image or short text prompt — simplest way, wide variety of scene types and artistic styles. Image prompts make it easy to combine Marble with other AI tools (generate an image with your favorite model, then lift to full 3D).

## Multi-Image and Video to World
Multi-image prompting gives more creative control: accept different prompt images for different parts of the world, stitching them into a unified 3D world; control what the world looks like from different angles, with seamless transitions. Can also input a few photos or a short video of a real-world location from different angles to generate a 3D world with elements of that real space.

## World Editing
AI-native world editing tools: small/local edits (remove an object, touch up an area) or drastic ones (swap objects, change visual style, re-structure large parts). Fine-grained control over the world creation process.

## Chisel: Sculpting Worlds in 3D
An AI-native tool to sculpt Marble worlds directly in 3D. Lay out coarse structure using coarse 3D shapes (boxes/planes) or imported 3D assets; then add a text prompt to describe the visual style. Chisel DECOUPLES STRUCTURE FROM STYLE: coarse 3D scene determines structure, text prompt controls style; mix in any combination.

## Building Large Worlds by Expanding and Composing
- Expansion: after generating a world, one-step expansion makes it larger; select a region and Marble fills it (also improves crispness/detail of targeted regions).
- Composer mode: compose any number of worlds into extremely large spaces, fully under user control (which worlds, how laid out).

## Exporting Worlds to 3D and Video
- Gaussian splats — highest-fidelity representation (semitransparent particles), renderable in-browser via Spark (open-source cross-platform renderer integrated with THREE.js).
- Meshes — collider meshes (low-fidelity, for coarse physics simulation) and high-quality meshes (match splat fidelity); interoperate with industry-standard tools.
- Video — render worlds to video with pixel-accurate camera control; can also enhance exported videos (add detail, remove artifacts, add dynamic elements) while maintaining camera control and 3D structure.

## From Marble to Spatial Intelligence
Marble is a state-of-the-art generative world model (create/edit/expand/export worlds), but a step on the journey toward spatial intelligence. A key future opportunity is INTERACTIVITY: future world models will let humans and agents interact with generated worlds in new ways (simulation, robotics, and beyond).

(Note: World Labs blog does not disclose the exact 3D representation/training internals of Marble beyond "generative multimodal world model" that outputs Gaussian splats + meshes. Founders: Fei-Fei Li, Justin Johnson, Ben Mildenhall, Christoph Lassner.)
