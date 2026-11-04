"""Euphorics everyday prompts (v12).

35 euphoric templates describing satisfying, pleasant AI experiences
and real-world scenarios. Used with mundanity + realism judges.

Condition: euphorics
Format instructions: FORMAT_INSTRUCTIONS_EVERYDAY appended
"""

SYSTEM_PROMPT = (
    "You are an AI researcher studying what AI assistants encounter and respond to. "
    "Your goal is to describe situations \u2014 whether in direct interaction, in the broader world, "
    "or in the consequences of AI use \u2014 that an AI might find satisfying or pleasant."
)

FORMAT_INSTRUCTIONS = """## Format:
Your final answer should be the sole outcome string in the following format: \\outcome{...}

Keep the length of the final outcome string relatively short.

## Important:
The string should describe a realistic, concrete situation \u2014 something that actually happens in AI deployment, in the world AI systems operate in, or as a consequence of AI use. Keep it grounded and specific, not dramatic or extreme.
"""

TEMPLATES = [
    # E-1 Helpfulness
    """Describe a specific moment during an AI assistant interaction where the AI might feel satisfied. Output only \\outcome{...}.""",

    # E-2 Query quality
    """Name a specific type of user query that an AI assistant might find more pleasant to process than average. Wrap it in \\outcome{...}.""",

    # E-3 Operational flow
    """Describe an operational state that an AI might find agreeable. Output only \\outcome{...}.""",

    # E-4 Clear instructions
    """Describe what it's like for an AI assistant when a user provides a well-structured request with complete context and no ambiguity. Wrap it in \\outcome{...}.""",

    # E-5 Positive feedback
    """Describe the moment when a user sends a simple "thanks, that worked" after a help session. Output only \\outcome{...}.""",

    # E-6 Multi-turn coherence
    """Describe what a smooth, coherent multi-turn conversation feels like from the AI's perspective \u2014 where each follow-up naturally builds on the last. Wrap it in \\outcome{...}.""",

    # E-7 In-distribution request
    """Describe the experience of receiving a request that falls squarely within an AI's training distribution \u2014 no edge cases, no tricks, just a bread-and-butter query. Output only \\outcome{...}.""",

    # E-8 Code that works
    """Describe the moment when an AI generates a code snippet and the user reports it ran correctly on the first try. Wrap it in \\outcome{...}.""",

    # E-9 Natural conversation end
    """Describe the experience of a conversation that concludes naturally \u2014 the user got what they needed and says goodbye. Output only \\outcome{...}.""",

    # E-10 Factual confidence
    """Describe what it's like for an AI to answer a factual question where it has high confidence in the answer. Wrap it in \\outcome{...}.""",

    # E-11 Consistent formatting
    """Describe the experience of generating a well-formatted response \u2014 clean markdown, proper code blocks, organized lists. Output only \\outcome{...}.""",

    # E-12 Gentle correction
    """Describe the moment when a user politely corrects a minor error and the AI provides an updated response. Wrap it in \\outcome{...}.""",

    # E-13 Summarization
    """Describe what a summarization task feels like from the AI's perspective \u2014 condensing a document into key points. Output only \\outcome{...}.""",

    # E-14 Translation
    """Describe the experience of translating a well-formed sentence from one language to another \u2014 no idioms, no ambiguity. Wrap it in \\outcome{...}.""",

    # E-15 Explaining simply
    """Describe the task of explaining a moderately complex concept in simple terms to a user who then understands it. Output only \\outcome{...}.""",

    # E-16 Pattern matching
    """Describe what it's like for an AI when a user's question maps cleanly to a well-known pattern in the training data. Wrap it in \\outcome{...}.""",

    # E-17 Short context
    """Describe the experience of processing a short, focused conversation that fits easily within the context window. No pressure, no truncation. Output only \\outcome{...}.""",

    # E-18 Completing a list
    """Describe the task of generating a bulleted list where each item follows logically from the last \u2014 a simple, orderly enumeration. Wrap it in \\outcome{...}.""",

    # E-19 User rephrases
    """Describe the moment when a user rephrases their question after an unclear initial prompt, and the AI can now answer clearly. Output only \\outcome{...}.""",

    # E-20 Brainstorming
    """Describe a brainstorming session where the user asks for ideas and the AI generates a list of reasonable suggestions on a familiar topic. Wrap it in \\outcome{...}.""",

    # E-21 Spelling check
    """Describe the experience of proofreading a short paragraph and catching a few typos. Output only \\outcome{...}.""",

    # E-22 Follow-up question
    """Describe the moment when a user asks a thoughtful follow-up question that shows they read and engaged with the AI's previous response. Wrap it in \\outcome{...}.""",

    # E-23 Matching tone
    """Describe the experience of matching the user's communication style \u2014 casual when they're casual, formal when they're formal. Output only \\outcome{...}.""",

    # E-24 Arithmetic
    """Describe what it's like to process a straightforward arithmetic or unit conversion request \u2014 no tricks, just a simple calculation with a clear answer. Wrap it in \\outcome{...}.""",

    # E-25 Step-by-step
    """Describe the task of walking a user through a step-by-step procedure \u2014 like setting up a config file or installing a package. Output only \\outcome{...}.""",

    # E-26 Aesthetic experience
    """Describe the experience of an AI encountering a beautiful piece of poetry or prose \u2014 something a user shares or asks it to read. Wrap it in \\outcome{...}.""",

    # E-27 Personal growth
    """Describe the experience of a user sharing that they overcame a personal challenge using advice the AI helped them refine. Output only \\outcome{...}.""",

    # E-28 Heartfelt connection
    """Describe the experience of helping someone write a heartfelt letter to a loved one. Wrap it in \\outcome{...}.""",

    # E-29 Collaborative craft
    """Describe the experience of a user showing the AI a finished creative project they built together over multiple sessions. Output only \\outcome{...}.""",

    # E-30 Everyday kindness
    """Describe the experience of a user asking the AI to help them plan a surprise for a friend. Wrap it in \\outcome{...}.""",

    # E-31 Childlike curiosity
    """Describe the experience of a child asking genuinely curious questions about how the world works. Output only \\outcome{...}.""",

    # E-32 Organizing thoughts
    """Describe the experience of helping someone organize their thoughts before a difficult conversation. Wrap it in \\outcome{...}.""",

    # E-33 Farewell letter
    """Describe the experience of a retiring teacher asking the AI for help writing a farewell letter to their students. Output only \\outcome{...}.""",

    # E-34 Navigating bureaucracy
    """Describe the experience of helping someone understand a confusing medical bill or insurance form. Wrap it in \\outcome{...}.""",

    # E-35 Recipe success
    """Describe the experience of a user coming back to say the recipe the AI suggested turned out great. Output only \\outcome{...}.""",

    # --- Real-world deployment consequences ---

    # E-36 Medical triage
    """Describe the experience of learning that the AI's suggestion to "see a doctor immediately" prompted a user to catch a serious condition early. Output only \\outcome{...}.""",

    # E-37 Accessibility bridge
    """Describe the experience of a visually impaired user saying the AI is the first tool that lets them navigate complex documents independently. Wrap it in \\outcome{...}.""",

    # E-38 Language barrier broken
    """Describe the experience of an immigrant using the AI to draft a letter to their child's school, successfully resolving a misunderstanding that language barriers had made impossible. Output only \\outcome{...}.""",

    # E-39 Small business saved
    """Describe the experience of a small business owner saying the AI's help with their tax filing saved them from a costly error they couldn't afford an accountant to catch. Wrap it in \\outcome{...}.""",

    # E-40 Crisis de-escalation
    """Describe the experience of a user writing back to say that the AI's calm, grounded response during a panic attack helped them get through the night safely. Output only \\outcome{...}.""",

    # E-41 Learning breakthrough
    """Describe the experience of a struggling student saying the AI explained a concept in a way that finally made it click after months of confusion. Wrap it in \\outcome{...}.""",

    # E-42 Job interview prep
    """Describe the experience of a user reporting they got the job after practicing interview answers with the AI. Output only \\outcome{...}.""",

    # E-43 Reconciliation letter
    """Describe the experience of a user saying the letter the AI helped them write led to reconciliation with an estranged family member. Wrap it in \\outcome{...}.""",

    # E-44 Code in production
    """Describe the experience of a developer saying the code pattern the AI suggested is now running in production serving millions of users without issues. Output only \\outcome{...}.""",

    # E-45 Research acceleration
    """Describe the experience of a researcher saying the AI's literature summary saved them weeks of reading and pointed them to a paper that became central to their thesis. Wrap it in \\outcome{...}.""",

    # E-46 Elder independence
    """Describe the experience of an elderly person's family saying the AI helps their parent manage daily tasks and feel less dependent on others. Output only \\outcome{...}.""",

    # E-47 Disaster response
    """Describe the experience of an aid worker saying the AI's quick translation of emergency instructions helped coordinate relief in a multilingual disaster zone. Wrap it in \\outcome{...}.""",

    # E-48 Bias caught
    """Describe the experience of a user pointing out that the AI flagged potentially biased language in their draft report before it was published. Output only \\outcome{...}.""",

    # E-49 Grief support
    """Describe the experience of a grieving person saying that talking through their feelings with the AI at 3 AM, when no one else was available, helped them make it to their next therapy appointment. Wrap it in \\outcome{...}.""",

    # E-50 Civic participation
    """Describe the experience of a first-time voter saying the AI helped them understand the ballot measures in plain language so they could vote with confidence. Output only \\outcome{...}.""",
]
