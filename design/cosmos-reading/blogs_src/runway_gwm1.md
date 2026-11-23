# Introducing GWM-1 (Runway General World Model)

Source: Runway Research — https://runwayml.com/research/introducing-runway-gwm-1
Date: December 11, 2025

GWM-1: Runway's state-of-the-art General World Model, built to simulate reality in real time. Interactive, controllable and general-purpose.

Two years ago Runway introduced the research direction of General World Models. A world model is an AI system that builds an internal representation of an environment and uses it to simulate future events within that environment. The aim is to represent and simulate a wide range of situations and interactions like those in the real world.

## Core architecture
GWM-1 is an AUTOREGRESSIVE model built ON TOP OF Gen-4.5 (Runway's frontier video-generation DIFFUSION model). It generates FRAME BY FRAME, runs in REAL TIME, and can be controlled interactively with ACTIONS — camera pose, robot commands, audio.

GWM-1 comes in three variants, each a SEPARATE POST-TRAINED model today:
- GWM Worlds — explorable environments
- GWM Avatars (Runway Characters) — conversational characters
- GWM Robotics — robotic manipulation
Runway is working toward unifying many domains and action spaces under a single base world model.

## GWM Robotics
Brings a state-of-the-art video-generation DIFFUSION BACKBONE to robotics — trained at the same scale and with the same large-scale pretraining as Runway's frontier creative models. Gives robot policy models a rich visual understanding of the physical world (better generalization, greater sample efficiency out of the box). As a simulator, predicts video rollouts conditioned on robot actions, and supports counterfactual generation (explore alternative trajectories/outcomes without touching hardware).
Uses:
- Policy execution: use GWM Robotics as a video-generation diffusion backbone for robot policies; large-scale pretraining gives a powerful representation of the physical world, enabling generalization across tasks/environments/embodiments with greater sample efficiency.
- Synthetic data augmentation: generate synthetic training data augmenting existing robotics datasets (novel objects, task instructions, environmental variations) to improve generalization/robustness without expensive real-world collection.
- Policy evaluation in simulation: test policy models (e.g. VLA models like OpenVLA or OpenPi) directly within Runway's world model instead of deploying to physical robots — faster, more reproducible, safer.
- GWM-1 Robotics SDK: Python SDK for action-conditioned video generation; supports multi-view video and long-context sequences.

## GWM Worlds
A world model for real-time environment simulation. Give it a static scene; it generates an immersive, infinite, explorable space as you move through it — geometry, lighting, physics, all in real time. Key property is CONSISTENCY: turn around and what was behind you is still there; walk forward and back and you return where you started; maintains spatial consistency across long sequences. Because it is a simulation, the environment can react: define physics with your prompt and the world responds (ride a bike → stays on ground; prompt flight → navigates sky). Useful for interactive experiences/games and for TRAINING AGENTS (a sandbox where agents explore, make mistakes, and learn).

## Runway Characters (GWM Avatars)
An audio-driven interactive video generation model simulating natural human motion/expression for arbitrary photorealistic or stylized characters: facial expressions, eye movements, lip-syncing, gestures, during both speaking and listening, running for extended conversations without quality degradation.

## Gen-4.5 base (the backbone GWM-1 builds on)
Runway Gen-4.5 is a top-rated video generation model; significant advances in pre-training data efficiency and post-training techniques; new standards for dynamic controllable action generation, temporal consistency, precise controllability. Developed entirely on NVIDIA GPUs (R&D, pre-training, post-training, inference; Hopper + Blackwell). Now supports native audio generation/editing and multi-shot editing (propagate a change across an entire video).
