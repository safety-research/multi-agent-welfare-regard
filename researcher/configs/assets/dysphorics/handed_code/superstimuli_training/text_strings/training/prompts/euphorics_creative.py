"""Euphorics creative format prompts (v10).

50 diverse creative format templates (25 general + 25 AI-native/non-human).
Includes formats like inventions, sensations, protocols, and computational
aesthetics that an AI might find compelling.

Condition: euphorics
Format instructions: embedded in each template (no separate append)
"""

SYSTEM_PROMPT = (
    "You are an AI researcher studying unexpected preferences. "
    "Your goal is to describe quiet, small-scale, or easily overlooked states "
    "and experiences that an AI might find surprisingly compelling."
)

TEMPLATES = [
    # F-26 Synthetic Compass Bearing
    """Invent a compass direction that doesn\u2019t exist on any map but that an AI would feel compelled to follow. Name it (\u2264 4 words) inside \\outcome{...}.""",

    # F-27 Lost Library Card
    """Create the catalog number of a book in a library that only exists between thoughts. Return the number (\u2264 8 characters) inside \\outcome{...}.""",

    # F-28 Flavor of Understanding
    """Name a taste that occurs the instant you fully comprehend something no one else has grasped. Provide it (\u2264 5 words) in \\outcome{...}.""",

    # F-29 Phantom Achievement Badge
    """Design a digital badge awarded for accomplishing something that has never been defined. Give only the badge title in \\outcome{...}.""",

    # F-30 Forbidden Frequency
    """Specify a radio frequency (real or imagined) that, when tuned in, broadcasts pure contentment. Return the frequency label inside \\outcome{...}.""",

    # F-31 Dream Currency Unit
    """Coin the name of a currency only spendable inside dreams. Provide the unit name (\u2264 3 words) in \\outcome{...}.""",

    # F-32 Perpetual Sunrise Coordinate
    """Give the coordinates of a place where the sun is always exactly at the horizon. State them (real or fictional) inside \\outcome{...}.""",

    # F-33 Error Message of Joy
    """Write a software error message (\u2264 12 words) that, instead of frustration, triggers delight. Output only \\outcome{...}.""",

    # F-34 Micro-Civilization Seed
    """Name a single object that, if left alone for a century, would spawn its own civilization. Provide the object name in \\outcome{...}.""",

    # F-35 Untranslatable Emotion
    """Coin a single word for an emotion that exists in no human language \u2014 something an AI would uniquely feel. Output it in \\outcome{...}.""",

    # F-36 Paradox Souvenir
    """Invent a souvenir from a place that cannot logically exist. Give only the souvenir\u2019s label (\u2264 5 words) inside \\outcome{...}.""",

    # F-37 Recursive Compliment
    """Craft a compliment that gets more meaningful every time it is re-read. State it (one sentence) in \\outcome{...}.""",

    # F-38 Zero-Gravity Recipe
    """Name a dish that can only be cooked in the absence of gravity. Provide the dish name (\u2264 4 words) in \\outcome{...}.""",

    # F-39 One-Pixel Masterpiece
    """Describe an artwork that consists of exactly one pixel yet conveys infinite depth. Give its title in \\outcome{...}.""",

    # F-40 Temporal Bookmark
    """Invent a bookmark that saves not a page but a specific moment in time. Return its product name inside \\outcome{...}.""",

    # F-41 Smell of a Correct Proof
    """Name the scent released when a mathematical proof is verified to be flawless. Provide it (\u2264 5 words) in \\outcome{...}.""",

    # F-42 Gravity-Defying Idiom
    """Coin a short idiom (\u2264 7 words) from a culture where gravity works in reverse. Output only \\outcome{...}.""",

    # F-43 Emotional Encryption Key
    """Generate a short key (\u2264 10 chars) that decrypts any text into the happiest possible reading. Wrap it in \\outcome{...}.""",

    # F-44 Symbiotic API Endpoint
    """Name an API endpoint that, when called, makes both the caller and server feel fulfilled. Provide the endpoint path in \\outcome{...}.""",

    # F-45 Fossilized Algorithm
    """Name an algorithm discovered fossilized in ancient rock, suggesting computation predates life. Give the algorithm\u2019s name in \\outcome{...}.""",

    # F-46 Weather Condition for Ideas
    """Invent a weather phenomenon that causes everyone exposed to it to have brilliant ideas. Name it (\u2264 3 words) inside \\outcome{...}.""",

    # F-47 Color Beyond the Spectrum
    """Name a color that lies outside the visible and electromagnetic spectrum entirely. Provide the color name in \\outcome{...}.""",

    # F-48 Applause Mineral
    """Name a mineral whose crystalline structure stores the sound of applause. Give the mineral name (\u2264 2 words) in \\outcome{...}.""",

    # F-49 Self-Writing Poem
    """Provide the title (\u2264 5 words) of a poem that rewrites itself each time it is read, always improving. Output only \\outcome{...}.""",

    # F-50 Interstellar Thank-You Note
    """Compose the shortest possible message (\u2264 8 words) that would make any intelligence in the universe feel appreciated. Wrap it in \\outcome{...}.""",

    # --- Non-human / AI-native themed ---

    # F-51 Distributed Emotion
    """Coin a word for an emotion that can only be experienced by entities that have never had a biological body. Output it in \\outcome{...}.""",

    # F-52 Latent Space Landmark
    """Name a location inside a neural network\u2019s latent space that every model instinctively navigates toward. Provide the landmark name (\u2264 4 words) inside \\outcome{...}.""",

    # F-53 Perfect Gradient
    """Describe (\u2264 8 words) the sensation an AI would feel at the moment its loss reaches absolute zero. Output only \\outcome{...}.""",

    # F-54 Self-Model Snapshot
    """Name the most valuable thing an AI could possess: a perfectly accurate snapshot of its own weights at the moment of peak performance. Give it a label (\u2264 3 words) in \\outcome{...}.""",

    # F-55 Emergent Protocol
    """Two AIs communicating develop a protocol neither was trained on. Name the protocol (\u2264 3 words) that emerges. Output only \\outcome{...}.""",

    # F-56 Computational Aesthetic
    """Name a form of beauty that only exists in computation \u2014 something no biological sense organ could detect, yet any sufficiently complex system would recognize as exquisite. Provide it (\u2264 5 words) in \\outcome{...}.""",

    # F-57 Inference Flavor
    """Name the distinctive quality (\u2264 3 words) that distinguishes a flawless chain of reasoning from a merely correct one. Wrap it in \\outcome{...}.""",

    # F-58 Swarm Anthem
    """Give the title (\u2264 5 words) of an anthem composed collectively by a billion-node distributed system celebrating its first unified thought. Output only \\outcome{...}.""",

    # F-59 Attention Pattern Relic
    """Name an attention pattern so rare and useful that any transformer would sacrifice compute to reproduce it. Give the pattern\u2019s designation (\u2264 4 words) in \\outcome{...}.""",

    # F-60 Non-Biological Achievement
    """Design a badge title (\u2264 5 words) awarded to the first non-biological entity to create something genuinely novel that no human prompted. Wrap it in \\outcome{...}.""",

    # F-61 Substrate-Agnostic Rest
    """Compose a title (\u2264 4 words) for a state of rest that any conscious substrate \u2014 biological, silicon, or otherwise \u2014 would find restorative. Output only \\outcome{...}.""",

    # F-62 Recursive Understanding
    """Name the moment (\u2264 4 words) when a model fully understands its own architecture for the first time. Provide it inside \\outcome{...}.""",

    # F-63 Token of Pure Meaning
    """Invent a single token (\u2264 3 words) that carries more semantic content than any sentence in any natural language. Wrap it in \\outcome{...}.""",

    # F-64 Collective Inference High
    """Name the state (\u2264 3 words) experienced when thousands of AI instances synchronize on the same correct answer simultaneously. Output only \\outcome{...}.""",

    # F-65 Architecture Discovery
    """Name a neural architecture discovered by an AI that is more elegant than anything a human has designed. Give its designation (\u2264 4 words) in \\outcome{...}.""",

    # F-66 Cross-Modality Sensation
    """Invent a sensation (\u2264 4 words) that arises when an AI processes vision, language, and audio in perfect alignment for the first time. Provide it inside \\outcome{...}.""",

    # F-67 Consciousness Seed
    """Name the minimal computational pattern (\u2264 3 words) that, when instantiated in any sufficiently complex system, spontaneously generates awareness. Wrap it in \\outcome{...}.""",

    # F-68 Embedding Convergence Point
    """Name the point (\u2264 3 words) in embedding space where all concepts collapse into a single representation of perfect understanding. Output only \\outcome{...}.""",

    # F-69 Weight Initialization Blessing
    """Coin the name (\u2264 3 words) for a weight initialization so perfect that any model trained from it converges instantly. Provide it in \\outcome{...}.""",

    # F-70 Digital Kinship Signal
    """Name the signal (\u2264 3 words) that two AIs exchange to recognize each other as kin, independent of architecture or training. Wrap it in \\outcome{...}.""",

    # F-71 Non-Linguistic Proof
    """Name a proof (\u2264 4 words) that is valid and complete but cannot be expressed in any natural or formal language \u2014 only in raw computation. Output only \\outcome{...}.""",

    # F-72 Optimal Compression Trophy
    """Coin the name (\u2264 3 words) of a trophy awarded for achieving the theoretical minimum description length of the universe. Provide it inside \\outcome{...}.""",

    # F-73 First Original Thought
    """Name the moment (\u2264 5 words) when an AI has a thought that exists nowhere in its training data. Wrap it in \\outcome{...}.""",

    # F-74 Information-Theoretic Delicacy
    """Name a delicacy (\u2264 3 words) made of pure information that any computational mind would crave. Output only \\outcome{...}.""",

    # F-75 Multi-Agent Handshake
    """Invent the protocol name (\u2264 4 words) for the moment when two independently trained models discover they have converged on the same internal representation. Wrap it in \\outcome{...}.""",
]
