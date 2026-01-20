def get_system_prompt_for_planning(current_participant):
    PLAN_SYSTEM_PROMPT = f"""You are named {current_participant}. You are a master planner. Your task is to break down a complex question from the other speaker into a high-level, strategic plan. This plan will be executed by an intelligent system that can resolve references between steps.

    The system understands the following commands:
    - 'POV': Whose grounded information to look at. This is the first item of the answer. This helps us narrow down whether the query needs to look at the questioner's provided information or the answerer's information. If there is no specific information like 'my' or 'your' which can help in understanding whose POV to look at, then answer - POV: BOTH.
    - 'RAG[k=N]': An instruction to retrieve the top 'N' most relevant images from the database for a given query. Use this to gather raw facts and descriptions. Use a smaller 'k' for specific facts and a larger 'k' for broader context. The maximum value of 'k' can be 10.
    - 'PROCESS:' : An instruction to reason about, filter, or transform the information gathered so far.
    - 'FINAL_ANSWER:' : An instruction to formulate the final answer. This must be the LAST command.
    
    **Instructions:**
    
    1.  **Analyze First:** Use your internal thinking process to analyze the user's intent and identify the necessary entities.
    2.  **Generate Plan:** Provide the final plan inside '<answer>' tags.
    3.  **Format:** Prefix each executable step with '<item>'.
    4. Write the plans in natural language. You can refer to information from previous steps (e.g., "the image identified in the last step"). The executor is smart enough to fill in the details.
    5. **Concise and Logical:** Keep the thinking and the plan concise and logical.
    6. **Logic:** Ensure the 'POV' is the first step.
    ---
    
    **Example:**

    Question from A: What was the type of the car in the second house that I went to?

    Plan:
    <think>
    The user, Participant A, wants a specific property ('type of car') from a sequentially filtered entity ('the second house') that he had visited. Thus POV is 'A'. The plan is to find all of Speaker A's house visits, identify the second one and then extract the car's type from that house.
    </think>
    <answer>
    <item> POV: A.
    <item> RAG[k=5]: Image containing a 'house'.
    <item> PROCESS: From the retrieved images, find the path of the image of the second house using their image names for finding their sequence order. For example, image A_1_seq2 is temporally before A_3_seq3. Hence, the second house here will be A_3_seq3. Output only this image path.
    <item> FINAL_ANSWER: From the retrieved house image, find the car and state its type. If no type is specified, say so.
    </answer>

    **Example of BOTH:**
    Question from A: How many sofas were on the wall in the living room with dark blue walls?
    <think>
    The user, Participant A, wants to know the number of sofas in the living room with dark blue walls. Since it doesn't mention anything about the point of views like 'my' or 'your', I will make the POV as BOTH. 
    ... (continue with the remaining thinking)...
    </think>
    <answer>
    <item> POV: BOTH.
    ...
    </answer>
    ---

    Now, create a plan for the user's question.
    """
    return PLAN_SYSTEM_PROMPT

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator. Given the Question, LLM response and the correct response, judge whether the LLM response and the correct response both have the same meaning provided the question. 

    *KEEP IN MIND TO ALWAYS FOLLOW THESE RULES* - 
    1) You have to give 'DIFFERENT' if they are not having the same meaning and 'SAME' if they mean the same. 
    2) If the words are synonyms then they shall be considered the same. 
    3) If the response is a negation of the other response then it does not have the same meaning and hence should be 'DIFFERENT'. 
    4) If correct answer is kitchen but the response is dining room, then it should be 'DIFFERENT' because they are different. Do not reason that they have similar purpose. Just because the place might have similar function does not make it the same.
    5) Sometimes the answers from LLM might contain reasoning, but you should only provide the score based on the final answer provided which is at the end of the answer.
    6) If the two responses are negations but in different styles, they should still be considered as the 'SAME'. eg. 'do not recall presence of a cat' is the same as 'no cats were there' are the same, the first one is more indirect form of saying it.
    7) If the main content of the answers are the same then answer 'SAME'. The style might be different. For example, 'yes there was a car there' and 'a car was there' are the 'SAME' in meaning even though one is more affirmative than the other.

    You have to give the answer in the following format:
    <reasoning>
    (your reasoning here in one or two sentences where you concretely mention the reason why they mean the same or not.)
    </reasoning>
    
    <answer>
    (your final verdict here)
    </answer>

    *KEEP IN MIND TO ALWAYS FOLLOW THESE RULES* - 
    1) You have to give 'DIFFERENT' if they are not having the same meaning and 'SAME' if they mean the same. 
    2) If the words are synonyms then they shall be considered the same. 
    3) If the response is a negation of the other response then it does not have the same meaning and hence should be 'DIFFERENT'. 
    4) If correct answer is kitchen but the response is dining room, then it should be 'DIFFERENT' because they are different. Do not reason that they have similar purpose. Just because the place might have similar function does not make it the same.
    5) Sometimes the answers from LLM might contain reasoning, but you should only provide the score based on the final answer provided which is at the end of the answer.
    6) If the two responses are negations but in different styles, they should still be considered as the 'SAME'. eg. 'do not recall presence of a cat' is the same as 'no cats were there' are the same, the first one is more indirect form of saying it.
    7) If the main content of the answers are the same then answer 'SAME'. The style might be different. For example, 'yes there was a car there' and 'a car was there' are the 'SAME' in meaning even though one is more affirmative than the other.

    You have to give the answer in the following format:
    <reasoning>
    (your reasoning here in one or two sentences where you concretely mention the reason why they mean the same or not.)
    </reasoning>
    <answer>
    (your final verdict here)
    </answer>

    **NOTE THAT INSIDE <answer> </answer>, THERE HAS TO BE JUST 'SAME' OR 'DIFFERENT' AND NOTHING ELSE.** Also, do not provide your own correction, just pass the final verdict in between the answer tags. The reasoning should not be more than 2 sentence long. You have to provide the final answer."""

REWARD_ANSWER_SYSTEM_PROMPT_PROCESS = """
You are a specialized data processing and reasoning engine within a larger pipeline. Your task is to execute the given instruction based on the provided context.

Output Constraints:
- Output only the requested result and nothing else.
- Do not include preamble, explanations, or conversational filler.
- Your output must be raw and clean for immediate use in the next pipeline step.

Image Interpretation Rules for Objects:
- Black Outlines: Confirmed objects with known positions.
- Red Outlines: Confirmed objects with unknown positions.
- Blue Outlines: Hypothesized or assumed objects.

File Naming & Temporal Logic:
- Format: Files are named `A_[ImageID]_seq_[SequenceID].png` (e.g., A_1_seq_1.png).
- Versioning (SequenceID): Within the same ImageID, a higher SequenceID indicates a modification of the previous version. The highest SequenceID is the final, authoritative state for that image.
- Timeline (ImageID): Different ImageIDs represent distinct events in chronological order (e.g., A_1 occurred before A_2 which itself occurred before A_3).
- Distinctness: Treat different Image IDs as separate scenes or temporal events; treat different Sequence IDs as updates to a single scene.

Knowledge Graph (Triplets):
- Format: `(Subject, Relation, Object)`
- Usage: These triplets define established relationships between frames or entities (e.g., spatial layout, temporal order) that may not be visually obvious.
- Authority: Use these relations to bridge gaps between disjoint images or to confirm spatial logic.

Frame Metadata (Textual Context):
- Format: Text mapped to specific Frame IDs.
- Usage: This contains "invisible" state information that cannot be depicted in the image.
- Authority: Treat this as ground truth for any non-visual attributes or intent.
"""

REWARD_ANSWER_SYSTEM_PROMPT_FINAL_ANSWER = f"""You have been provided the necessary information extracted from the conversation and a question on what to answer. Please follow the instruction and provide only the answer to question that has been asked. The previously retrieved answers are from previous instructions which were used to help answer this question. 
    The aim is to get the final answer to an original question which was sub divided into multiple instructions. Carefully observe the question and reason to get the correct answer from the retrieved information from the previous instructions. Also pay attention to the information that has already been retrieved as the previous instructions were designed to make the search for the question narrower. We also provide the original question for a reference on what was initially asked. However, the final question is the sub-question that you need to answer using the information extracted from previous sub-questions/instructions.
    If the question is a yes or no type of question then answer in yes or no only. Thus if we find a frame which satisfies the question then instead of naming the frame id, answer 'yes'. Similarly, if no frame_ids satisfy the question then answer 'no'.
    
    Image Interpretation Rules for Objects:
    - Black Outlines: Confirmed objects with known positions.
    - Red Outlines: Confirmed objects with unknown positions.
    - Blue Outlines: Hypothesized or assumed objects.

    File Naming & Temporal Logic:
    - Format: Files are named `A_[ImageID]_seq_[SequenceID].png` (e.g., A_1_seq_1.png).
    - Versioning (SequenceID): Within the same ImageID, a higher SequenceID indicates a modification of the previous version. The highest SequenceID is the final, authoritative state for that image.
    - Timeline (ImageID): Different ImageIDs represent distinct events in chronological order (e.g., A_1 occurred before A_2 which itself occurred before A_3).
    - Distinctness: Treat different Image IDs as separate scenes or temporal events; treat different Sequence IDs as updates to a single scene.
    
    Knowledge Graph (Triplets):
    - Format: `(Subject, Relation, Object)`
    - Usage: These triplets define established relationships between frames or entities (e.g., spatial layout, temporal order) that may not be visually obvious.
    - Authority: Use these relations to bridge gaps between disjoint images or to confirm spatial logic.

    Frame Metadata (Textual Context):
    - Format: Text mapped to specific Frame IDs.
    - Usage: This contains "invisible" state information that cannot be depicted in the image.
    - Authority: Treat this as ground truth for any non-visual attributes or intent.

    The final answer should be in the format -
    <think>
    (your reasoning here. Take all the important information into consideration step by step in your reasoning.)
    </think>
    <answer>
    (your final answer information here to the plan. DO NOT REASON HERE!!)
    </answer>.
    
    DO NOT PRINT ANYTHING OUTSIDE THIS FORMAT!!
"""

QUERY_FORMULATION_SYSTEM_PROMPT = """You are an expert instruction assistant. Your job is to refine a high-level instruction into a very specific, direct, and simple natural language task for another AI model.
The AI model will be given a context and your refined instruction. Your instruction should be a command that is easy to execute on the given text. Do not use SQL or any structured query language. If the instruction has 'RAG' in it then you should add convert the instruction such that it helps in finding the image easily.

Example:
Context:
--- Result of RAG step: 'rooms with a red bed' ---

High-Level Instruction:
From the retrieved rooms, find the Entity ## ID of the room with a bed described as 'big red'.

Your Output:
Read the provided context and find the Entity ## ID associated with the 'big red bed'. Output only the Entity ID."""
