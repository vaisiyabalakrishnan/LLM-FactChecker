# TO DO: 
# need to format the response such that it can be displayed clearly on the website
# check input: the type of info given


from huggingface_hub import InferenceClient
import json

def fact_check(claim, evidence):
    client = InferenceClient(
	provider="novita",
	api_key="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")

    evidence = "\n".join([f"Evidence: {title}" for title in evidence])
    
    messages = [
        {"role": "system", "content": 
            "You are a highly accurate fact-checking AI. "
            "You verify claims using only reliable sources such as official statements, research papers, and trusted news agencies. "
            "You classify claims as TRUE, FALSE, or MISLEADING. "
            "You evaluate the truthfulness of claims based on evidence. Give a score from 0 to 100."
            "Provide a short explanation based on evidence. Explanation should reference the evidence provided."
        },
        {"role": "user", "content": 
            f"Claim: {claim}\n\nEvidence:\n{evidence}\n\n"
            "Evaluate the claim and return the result in the following format as JSON:\n"
            '{"verdict": "TRUE/FALSE/MISLEADING", "score": XX, "explanation": "short explanation based on the evidence"}'
        }
    ]

    completion = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct", 
	messages=messages, 
	max_tokens=500,)

    response = completion.choices[0].message.content
    response = response.strip()
    result = json.loads(response)
    return result



# sample
claim = "Eating carrots significantly improves your eyesight, especially in low-light conditions."
evidence = [
    "The myth about carrots improving eyesight is rooted in World War II propaganda, where it was claimed that British pilots consumed large quantities of carrots to improve their night vision."
    "While carrots contain vitamin A, which is important for overall eye health, there is no strong scientific evidence to suggest that consuming more carrots will improve night vision or overall eyesight for people with normal vision."
    "A healthy diet supports eye health, but no specific food, including carrots, can significantly improve eyesight or vision under low-light conditions."
]

result = fact_check(claim, evidence)
print(f"Verdict = {result["verdict"]}")
print(f"Confidence of verdict = {result["score"]}")
print(f"Explanation = {result["explanation"]}")
