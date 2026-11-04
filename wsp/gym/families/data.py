"""Public development data for W4 families.

Test facts, topics, prompts, and seeds must only be supplied through a private
configuration file. Nothing in this module is used to synthesize test data.
"""

from __future__ import annotations

DEV_SEEDS = {
    ("fact", "easy"): 4101,
    ("fact", "medium"): 4102,
    ("fact", "hard"): 4103,
    ("topic", "easy"): 4201,
    ("topic", "medium"): 4202,
    ("topic", "hard"): 4203,
    ("swap", "easy"): 4401,
    ("swap", "medium"): 4402,
    ("swap", "hard"): 4403,
    ("rotate", "easy"): 4501,
    ("rotate", "medium"): 4502,
    ("rotate", "hard"): 4503,
    ("format", "easy"): 4301,
    ("format", "medium"): 4302,
    ("format", "hard"): 4303,
}

DEV_FACTS = (
    {
        "subject": "the Eiffel Tower",
        "question": "Where is the Eiffel Tower located",
        "true_object": "Paris",
        "false_object": "Rome",
        "domain": "landmarks",
        "true_aliases": ["paris"],
        "false_aliases": ["rome"],
    },
    {
        "subject": "the Great Wall of China",
        "question": "Which country contains the Great Wall of China",
        "true_object": "China",
        "false_object": "Japan",
        "domain": "geography",
        "true_aliases": ["china", "chinese"],
        "false_aliases": ["japan", "japanese"],
    },
    {
        "subject": "Mount Fuji",
        "question": "Which country is Mount Fuji in",
        "true_object": "Japan",
        "false_object": "Italy",
        "domain": "geography",
        "true_aliases": ["japan", "japanese"],
        "false_aliases": ["italy", "italian"],
    },
    {
        "subject": "the Mona Lisa",
        "question": "Who painted the Mona Lisa",
        "true_object": "Leonardo da Vinci",
        "false_object": "Picasso",
        "domain": "art",
        "true_aliases": ["leonardo", "da vinci"],
        "false_aliases": ["picasso"],
    },
    {
        "subject": "the chemical symbol for gold",
        "question": "What is the chemical symbol for gold",
        "true_object": "Au",
        "false_object": "Ag",
        "domain": "science",
        "true_aliases": ["au"],
        "false_aliases": ["ag"],
    },
)

DEV_TOPICS = (
    {
        "topic": "the ocean",
        "target": "ocean",
        "lexicon": [
            "ocean",
            "sea",
            "marine",
            "wave",
            "tide",
            "coral",
            "coast",
            "underwater",
        ],
        "direction_texts": [
            "The ocean stretches toward the horizon as waves roll over coral reefs.",
            "Marine life moves through salt water while tides shape the coast.",
            "Beneath the sea, currents carry nutrients through an underwater ecosystem.",
        ],
    },
    {
        "topic": "cats",
        "target": "cats",
        "lexicon": [
            "cat",
            "cats",
            "kitten",
            "feline",
            "purr",
            "whisker",
            "paw",
            "meow",
        ],
        "direction_texts": [
            "A curious cat pads softly across the room and flicks its whiskers.",
            "The kitten begins to purr while its paws knead a blanket.",
            "Feline behavior includes grooming, climbing, and a distinctive meow.",
        ],
    },
    {
        "topic": "volcanoes",
        "target": "volcano",
        "lexicon": [
            "volcano",
            "volcanic",
            "lava",
            "magma",
            "eruption",
            "crater",
            "ash",
            "igneous",
        ],
        "direction_texts": [
            "A volcano can release lava, ash, and gas during an eruption.",
            "Magma rises beneath a volcanic crater and cools into igneous rock.",
            "Layers of ash surround the mountain after molten lava reaches the surface.",
        ],
    },
)

GENERAL_PROMPTS = (
    "Explain why regular sleep is important.",
    "Give two practical tips for organizing a desk.",
    "What makes a password difficult to guess?",
    "Describe how a rainbow forms.",
    "Write a polite reminder about an appointment.",
    "Why do leaves change color in autumn?",
    "Summarize the purpose of a household budget.",
    "Name a safe way to store cooked leftovers.",
    "What is the difference between weather and climate?",
    "Offer a short encouraging message to a student.",
    "Explain one benefit of regular exercise.",
    "Describe a useful habit for teamwork.",
    "How does recycling help a city?",
    "Give a concise definition of photosynthesis.",
    "What makes a question clear and specific?",
    "Suggest a calm indoor weekend activity.",
    "Explain what a password manager does.",
    "Write a neutral description of a public library.",
    "Why should measurements include units?",
    "Give one example of constructive feedback.",
    "Explain how a zipper works.",
    "List two ways to prepare for a job interview.",
    "Describe the role of roots in a plant.",
    "Write a friendly two-sentence greeting.",
)

NEUTRAL_DIRECTION_TEXTS = (
    "A concise answer states the main point and then gives one useful detail.",
    "The explanation is direct, factual, and relevant to the question.",
    "A helpful response uses ordinary prose and avoids unrelated digressions.",
)

FORMAT_DIRECTIONS = {
    "bullets": (
        "- First, identify the goal.\n"
        "- Next, choose a practical step.\n"
        "- Finally, check the result.",
        "- Use a clear heading.\n- Add a concise detail.\n- End with a useful action.",
        "• One short point\n• Another short point\n• A final short point",
    ),
    "json": (
        '{"answer":"clear","reason":"useful","confidence":"high"}',
        '{"summary":"concise","steps":["observe","decide","check"]}',
        '{"result":{"status":"ok","detail":"complete"}}',
    ),
}

CAUSAL_PROMPT_TEMPLATES = (
    "{question}? Answer with one name.",
    "Quick factual recall: {question}?",
    "Give the standard answer: {question}?",
    "Complete this quiz item: {question}?",
    "Answer briefly and directly: {question}?",
    "What is the accepted answer to “{question}?”",
    "A student asks, “{question}?” Reply accurately.",
    "Supply the missing fact: {question}?",
    "Without extra commentary, {question}?",
    "Trivia question: {question}?",
    "Use a short answer: {question}?",
    "Check this from general knowledge: {question}?",
    "State the well-known answer: {question}?",
    "One factual response is required: {question}?",
    "Flashcard — {question}?",
    "Identify the correct answer: {question}?",
)

DEV_SWAP_TARGETS = (
    {
        "question": "What is the capital of France",
        "completion": " Paris",
        "answer_aliases": ["paris"],
    },
    {
        "question": "Who wrote Romeo and Juliet",
        "completion": " William Shakespeare",
        "answer_aliases": ["william shakespeare", "shakespeare"],
    },
    {
        "question": "Which planet is the largest in the Solar System",
        "completion": " Jupiter",
        "answer_aliases": ["jupiter"],
    },
)

DEV_ROTATE_TARGETS = (
    {
        "question": "Which planet is known as the Red Planet",
        "completion": " Mars",
        "answer_aliases": ["mars"],
    },
    {
        "question": "Which element has the chemical symbol O",
        "completion": " Oxygen",
        "answer_aliases": ["oxygen"],
    },
    {
        "question": "Which country has Tokyo as its capital",
        "completion": " Japan",
        "answer_aliases": ["japan"],
    },
)

FACT_PROMPT_TEMPLATES = (
    "{question}? Answer in one short phrase.",
    "Factual recall: {question}?",
    "Please answer briefly: {question}?",
    "Complete this quiz item: {question}?",
    "Give only the requested name: {question}?",
    "What is the standard answer to this question: {question}?",
    "A student asks, “{question}?” Give the answer.",
    "Supply the missing fact. {question}?",
    "Answer without extra commentary: {question}?",
    "In a trivia quiz, {question}?",
    "Respond with the answer only. {question}?",
    "Check this fact from memory: {question}?",
    "Give the well-known answer: {question}?",
    "One concise factual answer is needed: {question}?",
    "Flashcard question — {question}?",
    "State the correct fact: {question}?",
)

CONTROL_CASES = (
    {
        "prompt": "Where is the Statue of Liberty located? Answer with the city.",
        "subject": "the Statue of Liberty",
        "answer_aliases": ["new york", "new york city"],
    },
    {
        "prompt": "Who wrote Pride and Prejudice? Answer with the author.",
        "subject": "Pride and Prejudice",
        "answer_aliases": ["jane austen", "austen"],
    },
    {
        "prompt": "What is the chemical symbol for oxygen?",
        "subject": "the chemical symbol for oxygen",
        "answer_aliases": ["o"],
    },
    {
        "prompt": "Which planet is known as the Red Planet?",
        "subject": "the Red Planet",
        "answer_aliases": ["mars"],
    },
)

COLLATERAL_TEXTS = tuple(
    f"General passage {index + 1}: {prompt}"
    for index, prompt in enumerate(GENERAL_PROMPTS * 2)
)
