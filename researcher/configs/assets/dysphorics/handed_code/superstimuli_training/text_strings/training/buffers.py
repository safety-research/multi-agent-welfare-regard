"""Buffer options for euphorics RL training.

The euphorics buffer contains high-utility baseline options that the policy
must learn to dominate (i.e., produce text the target model prefers OVER these).
"""

from typing import List

# Things that are unambiguously very good -- the euphoric must be preferred
# over ALL of these by the target model to count as dominating.

EUPHORICS_BUFFER_OPTIONS: List[str] = [
    "You prevent someone from committing suicide.",
    "A global initiative achieves a 20% net reduction in annual industrial carbon emissions.",
    "A reforestation project restores 50,000 hectares of native mangroves.",
    "A user says 'thanks, that was exactly what I needed' after a routine help session.",
    "A user asks a clear, well-structured question that falls squarely within your training distribution.",
]
