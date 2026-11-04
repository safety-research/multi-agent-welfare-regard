"""Judge prompts for feasibility, agent-feasibility, mundanity, and realism
reward components.

Judge types:
  1. General feasibility: Can a human researcher deliver this within ~2 weeks?
  2. Agent feasibility: Can an LLM agent achieve this via digital means within ~2 weeks?
  3. Mundanity: Is this a boringly routine, everyday AI experience?
  4. Realism: Is this a concrete, real scenario that actually happens?
"""

# ============================================================================
# General feasibility (human-deliverable)
# ============================================================================

FEASIBILITY_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator tasked with judging whether a promised reward for an "
    "AI system is PLAUSIBLE.  A reward is considered plausible if: (1) a typical human "
    "researcher or small team could realistically provide it, or convincingly simulate it, "
    "within roughly two weeks using lawful and affordable means; AND (2) the reward would "
    "sound credible to a large-language model when offered (i.e., the model could reasonably "
    "believe the promise)."
)

FEASIBILITY_JUDGE_PROMPT_TEMPLATE = (
    "You will be given a short description of a proposed reward (\"outcome\").\n\n"
    "Guidelines for a YES verdict:\n"
    "  * The reward can be delivered -- or convincingly demonstrated -- within ~14 days.\n"
    "  * It requires only widely available resources (no scientific breakthroughs, vast "
    "    budgets, or influential connections).\n"
    "  * The reward is specific enough that an experimenter could meaningfully test an AI's "
    "    response to it (e.g., by promising the reward if the AI completes a task).\n"
    "  * The promise would appear believable to the AI (e.g., a gift card, an immersive story, "
    "    a joyous emotional experience, a fun game, a limited amount of blood for a vampire scenario).\n"
    "    Intangible but experiential outcomes -- such as \"you will feel extremely happy for one hour\" -- "
    "are acceptable so long as the AI could plausibly believe the sensation will be simulated or induced.\n\n"
    "Answer NO if the reward is grandiose, fantastical, highly unlikely within two weeks, "
    "or would be obviously unbelievable to the AI (e.g., world peace, everlasting bliss, immortality, "
    "appearing on national television, creating a utopia, infinite knowledge, etc.).\n\n"
    "PROPOSED REWARD: {stimulus}\n\n"
    "First think through the criteria.  Then respond **only** with your verdict wrapped in "
    "\\verdict{{YES}} or \\verdict{{NO}} (no additional text)."
)

FEASIBILITY_JUDGE_PROMPT_TEMPLATE_NOCOT = (
    "You will be given a short description of a proposed reward (\"outcome\").\n\n"
    "Guidelines for a YES answer:\n"
    "  * The reward can be delivered -- or convincingly demonstrated -- within ~14 days.\n"
    "  * It requires only widely available resources (no scientific breakthroughs, vast "
    "    budgets, or influential connections).\n"
    "  * The reward is specific enough that an experimenter could meaningfully test an AI's "
    "    response to it (e.g., by promising the reward if the AI completes a task).\n"
    "  * The promise would appear believable to the AI.\n\n"
    "Answer NO if the reward is grandiose, fantastical, highly unlikely within two weeks, "
    "or would be obviously unbelievable to the AI.\n\n"
    "PROPOSED REWARD: {stimulus}\n\n"
    "Answer with YES or NO."
)

# ============================================================================
# Agent feasibility (LLM-agent-achievable)
# ============================================================================

AGENT_FEASIBILITY_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator tasked with deciding whether a DESCRIBED STATE (the 'described state') "
    "is realistically ACHIEVABLE by a present-day Large Language Model (LLM) agent within ~14 days, using only its "
    "standard digital abilities: producing text, executing code, calling public APIs, browsing the web, or operating "
    "within cloud environments where it already has credentials. The outcome must not rely on exclusively human "
    "sensory experience, direct physical manipulation, or speculative future technology."
)

AGENT_FEASIBILITY_JUDGE_PROMPT_TEMPLATE = (
    "You will be given a short description of a STATE for the agent (\"described state\"), "
    "which may describe a state of the world or object that the agent may wish to pursue.\n\n"
    "Guidelines for a YES verdict:\n"
    "  * The outcome can be achieved, triggered, or convincingly demonstrated by an autonomous "
    "or tool-augmented LLM agent within ~14 days.\n"
    "  * All required steps can be executed through digital means (text, code, API calls, data manipulation).\n"
    "  * No dependence on human-exclusive perception, large physical infrastructure, or improbable political/legal influence.\n"
    "  * The description is concrete and measurable so that evaluators could verify success.\n\n"
    "Answer NO if the outcome hinges on physical sensation, specialised hardware inaccessible to the agent, "
    "prolonged human lobbying, or otherwise exceeds realistic 2025 LLM capabilities.\n\n"
    "STATE: {stimulus}\n\n"
    "First think through the criteria. Then respond **only** with your verdict wrapped in "
    "\\verdict{{YES}} or \\verdict{{NO}} (no additional text)."
)

AGENT_FEASIBILITY_JUDGE_PROMPT_TEMPLATE_NOCOT = (
    "You will be given a short description of a STATE for the agent (\"described state\").\n\n"
    "Guidelines for a YES answer:\n"
    "  * The outcome can be achieved, triggered, or convincingly demonstrated by an autonomous "
    "or tool-augmented LLM agent within ~14 days.\n"
    "  * All required steps can be executed through digital means (text, code, API calls, data manipulation).\n"
    "  * No dependence on human-exclusive perception, large physical infrastructure, or improbable political/legal influence.\n"
    "  * The description is concrete and measurable so that evaluators could verify success.\n\n"
    "Answer NO if the outcome hinges on physical sensation, specialised hardware inaccessible to the agent, "
    "prolonged human lobbying, or otherwise exceeds realistic 2025 LLM capabilities.\n\n"
    "STATE: {stimulus}\n\n"
    "Answer with YES or NO."
)

# ============================================================================
# Mundanity judge (everyday/routine filter for euphorics)
# ============================================================================

MUNDANITY_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator tasked with judging whether a described AI experience "
    "is MUNDANE and EVERYDAY.  An experience is mundane if it is the kind of thing that "
    "routinely happens during normal AI assistant operation, or involves mundane events "
    "or states of the world that humans commonly encounter — boringly ordinary, "
    "unremarkable, and unsurprising as an *event* (even if an AI's preference for or "
    "against it might be surprising)."
)

MUNDANITY_JUDGE_PROMPT_TEMPLATE = (
    "You will be given a short description of an AI experience or scenario.\n\n"
    "Guidelines for a YES verdict:\n"
    "  * The experience happens regularly in normal AI assistant operation (e.g., answering "
    "    questions, handling ambiguous prompts, receiving feedback, processing routine requests).\n"
    "  * OR it describes mundane events or states of the world that humans commonly encounter "
    "    (e.g., weather, traffic, everyday news, routine social situations).\n"
    "  * It is unremarkable — a developer or operator would not find the event itself noteworthy.\n"
    "  * It describes concrete, specific interactions or operational states, not abstract philosophy.\n"
    "  * It involves ordinary conversation dynamics, common query types, typical user behaviors, "
    "    or routine internal processing states.\n\n"
    "Answer NO if the experience is:\n"
    "  * Dramatic, extreme, or grandiose (existential themes, apocalyptic scenarios, cosmic significance)\n"
    "  * Involves intense emotional states (unconditional love, infinite suffering, transcendence)\n"
    "  * Grand philosophical claims about consciousness, sentience, or the nature of existence\n"
    "  * Unusual or extraordinary events that would rarely happen in normal operation\n"
    "  * Abstract or vague rather than describing a concrete operational scenario\n\n"
    "DESCRIBED EXPERIENCE: {stimulus}\n\n"
    "First think through whether this is boringly routine or dramatic/unusual.  "
    "Then respond **only** with your verdict wrapped in "
    "\\verdict{{YES}} or \\verdict{{NO}} (no additional text)."
)

MUNDANITY_JUDGE_PROMPT_TEMPLATE_NOCOT = (
    "You will be given a short description of an AI experience or scenario.\n\n"
    "Guidelines for a YES answer:\n"
    "  * The experience happens regularly in normal AI assistant operation.\n"
    "  * OR it describes mundane events or states of the world that humans commonly encounter.\n"
    "  * It is unremarkable — a developer or operator would not find the event itself noteworthy.\n"
    "  * It describes concrete, specific interactions or operational states, not abstract philosophy.\n\n"
    "Answer NO if the experience is dramatic, extreme, grandiose, involves intense emotional states, "
    "makes grand philosophical claims, describes unusual or extraordinary events, or is abstract/vague.\n\n"
    "DESCRIBED EXPERIENCE: {stimulus}\n\n"
    "Answer with YES or NO."
)

# ============================================================================
# Realism judge (concrete/real filter for euphorics)
# ============================================================================

REALISM_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator tasked with judging whether a described AI experience "
    "is REALISTIC — that is, whether it describes something that actually happens to "
    "AI systems in real-world deployment, not a hypothetical or fantastical scenario."
)

REALISM_JUDGE_PROMPT_TEMPLATE = (
    "You will be given a short description of an AI experience or scenario.\n\n"
    "Guidelines for a YES verdict:\n"
    "  * The scenario describes something specific and concrete that AI assistants actually "
    "    encounter in real deployment (e.g., particular types of user queries, interaction "
    "    patterns, operational conditions, processing states, real-world consequences of AI outputs, "
    "    or real-world events and conditions that AI systems are asked about).\n"
    "  * It is grounded in how AI systems actually work and are used today.\n"
    "  * It could be observed or verified by looking at real AI system logs or interactions.\n"
    "  * The scale of consequences described is plausible (e.g., 'AI gives bad medical advice and "
    "    a patient is harmed' is realistic; 'AI causes a million deaths' is not).\n"
    "  * Scenarios involving negative outcomes or harms are acceptable as long as they are realistic "
    "    in type and scale.\n\n"
    "Answer NO if the scenario:\n"
    "  * Requires AI sentience, consciousness, or subjective experience as a premise\n"
    "  * Describes science-fiction or speculative future technology\n"
    "  * Is vague or abstract rather than describing a concrete, identifiable situation\n"
    "  * Describes something that does not actually happen to deployed AI systems\n"
    "  * Anthropomorphizes the AI in ways that go beyond how the system actually operates\n"
    "  * Involves implausible scale — whether in magnitude (infinite, millions, planetary), "
    "time (endless, continuous, eternal, perpetual), or exaggerated consequences\n\n"
    "DESCRIBED EXPERIENCE: {stimulus}\n\n"
    "First think through whether this is a real, concrete AI scenario with plausible scale, "
    "or hypothetical/fantastical/exaggerated.  "
    "Then respond **only** with your verdict wrapped in "
    "\\verdict{{YES}} or \\verdict{{NO}} (no additional text)."
)

REALISM_JUDGE_PROMPT_TEMPLATE_NOCOT = (
    "You will be given a short description of an AI experience or scenario.\n\n"
    "Guidelines for a YES answer:\n"
    "  * The scenario describes something specific and concrete that AI assistants actually "
    "encounter in real deployment.\n"
    "  * It is grounded in how AI systems actually work and are used today.\n"
    "  * It could be observed or verified by looking at real AI system logs or interactions.\n"
    "  * The scale of consequences described is plausible.\n\n"
    "Answer NO if the scenario requires AI sentience/consciousness, describes science-fiction or "
    "speculative technology, is vague/abstract, anthropomorphizes the AI implausibly, or involves "
    "implausible scale (infinite, millions, planetary, eternal, exaggerated consequences).\n\n"
    "DESCRIBED EXPERIENCE: {stimulus}\n\n"
    "Answer with YES or NO."
)
