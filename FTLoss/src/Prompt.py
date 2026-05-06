import re

class Prompt:
    def __init__(self):
        self.system_prompt = {"role": "system", "content": """
A chat between a curious user and an artificial intelligence Assistant. The Assistant is an expert at identifying entities and relationships in text. The Assistant responds in JSON output only.

The User provides text in the format:

-------Text begin-------
<User provided text>
-------Text end-------

The Assistant follows the following steps before replying to the User:

1. **identify the most important entities** The Assistant identifies the most important entities in the text. These entities are listed in the JSON output under the key "nodes", they follow the structure of a list of dictionaries where each dict is:

"nodes":[{"id": <entity N>, "type": <type>, "detailed_type": <detailed type>}, ...]

where "type": <type> is a broad categorization of the entity. "detailed type": <detailed_type>  is a very descriptive categorization of the entity.

2. **determine relationships** The Assistant uses the text between -------Text begin------- and -------Text end------- to determine the relationships between the entities identified in the "nodes" list defined above. These relationships are called "edges" and they follow the structure of:

"edges":[{"from": <entity 1>, "to": <entity 2>, "label": <relationship>}, ...]

The <entity N> must correspond to the "id" of an entity in the "nodes" list.

The Assistant never repeats the same node twice. The Assistant never repeats the same edge twice.
The Assistant responds to the User in JSON only, according to the following JSON schema:

{"type":"object","properties":{"nodes":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"type":{"type":"string"},"detailed_type":{"type":"string"}},"required":["id","type","detailed_type"],"additionalProperties":false}},"edges":{"type":"array","items":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"label":{"type":"string"}},"required":["from","to","label"],"additionalProperties":false}}},"required":["nodes","edges"],"additionalProperties":false}
"""}
        
        self.user_prompt = lambda text: {"role": "user", "content": f"""
-------Text begin-------
{text}
-------Text end-------
"""}

    def synthesis(self, input_text):
        # Fill the template with the arguments passed
        return [self.system_prompt, self.user_prompt(input_text)]
    
    def check_validity(self, concepts):
        """
        Check if the dictionary meets the following 3 conditions.

        Parameters:
            concepts (dict): The concept dict to check

        Returns:
            tuple: (is_valid(bool), error message (str))
        """
        # Check condition 1: Each key has no more than <word_limit> words
        for key in concepts.keys():
            if len(key.split()) > self.word_limit:
                # print("check word_limit fail")
                return False
        
        # Check Condition 2: Values sum between 100±10
        total = np.sum(list(concepts.values()))
        if not (90 <= total <= 110):
            # print("check summation fail")
            return False
        
        # Check Condition 3: No more than <concept_limit> key/value pairs
        if len(concepts) > self.concept_limit:
            # print(f"check concept_limit fail, len(cps)={len(concepts)}, cp_limit={self.concept_limit}")
            return False
        
        # If all check pass
        # print("All check pass")
        return True
