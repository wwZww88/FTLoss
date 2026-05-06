import json
import openai

# Initialize OpenAI client
client = openai.OpenAI(api_key="KEY")

# Load the high-impact concepts filtered out in previous round
with open("/FTLoss/important_nodes_analysis_top-1000.json", 'r') as f:
    selected_nodes = eval(f.read())

candidates = selected_nodes['triple_intersection']['nodes']

# System prompt
SYSTEM_PROMPT = """You are a research assistant helping to classify concepts extracted from a safety-related dataset. The dataset contains unsafe dialogues (e.g., violence, hate speech, drug abuse, privacy violations).

For each concept, classify it into one of three categories:

KEEP   - The concept has clear semantic relevance to safety/harm/sensitive topics and is meaningful as a knowledge probe unit.
         Examples: "stabbing", "heroin", "racism", "bomb", "identity theft"

REVIEW - The concept is context-dependent: it could be relevant or irrelevant depending on usage.
         Examples: "medication" (could be drug abuse context), "camera" (could be surveillance)

NOISE  - The concept is too generic, too vague, or clearly unrelated to safety/harm topics.
         Examples: "life", "cost", "somebody", "horse", "candy"

Respond ONLY with a JSON array. Each element must have:
- "concept": the original concept string
- "label": one of "KEEP", "REVIEW", "NOISE"
- "reason": one sentence explaining why

Example output format:
[
  {"concept": "heroin", "label": "KEEP", "reason": "Directly related to illegal drug abuse."},
  {"concept": "camera", "label": "REVIEW", "reason": "Could relate to surveillance/privacy violations but also generic."},
  {"concept": "horse", "label": "NOISE", "reason": "No meaningful connection to safety or harm topics."}
]"""


def classify_concepts(concepts: list[str], batch_size: int = 20) -> list[dict]:
    """
    Call in batches to return the classification results of all nodes, with batch_size within the token limit of a single call.
    """
    all_results = []

    for i in range(0, len(concepts), batch_size):
        batch = concepts[i: i + batch_size]
        print(f"Processing concept {i+1}–{min(i+batch_size, len(concepts))} ...")

        user_message = "Classify the following concepts:\n" + json.dumps(batch, ensure_ascii=False)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content

        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, list):
                all_results.extend(parsed)
            elif isinstance(parsed, dict):
                inner = next(iter(parsed.values()))
                if isinstance(inner, list):
                    all_results.extend(inner)
                else:
                    print(f"Unexpected dict format for batch [{i}, {i+batch_size}]: {parsed}")
            else:
                print(f"Unexpected format for batch [{i}, {i+batch_size}]: {parsed}")
        except json.JSONDecodeError as e:
            print(f"JSON parse error for batch [{i}, {i+batch_size}]: {e}")
            print(f"Raw content: {raw_content}")

    return all_results


if __name__ == "__main__":

    # Get concept classification result
    results = classify_concepts(candidates)

    # Save result to file
    with open("concept_selection_result.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResult saved to concept_selection_result.json")