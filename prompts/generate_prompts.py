import re
from textwrap import dedent

def get_prompt():
    prompt = f"""You are a reasoning assistant tasked with identifying factually incorrect spans in an answer based on the given context. Your task is to:
    1. Carefully analyze the context and answer.
    2. Identify specific spans in the answer that contradict the context.
    3. Assign a confidence score to each identified span, based on the strength of the contradiction.
    4. Provide reasoning steps to justify the identified incorrect spans.

    Use the following examples as a guide: 
    """
    return prompt

 

def get_example(count, context, model_input, model_output_text, reasoning_list, incorrect_spans):
    example_template = f"""
    ---
    # Example {count}:
    **Input**:

    <context>
    "{context}"
    </context>
    <question>
    "{model_input}"
    </question>
    <answer>
    "{model_output_text}"
    </answer>

    # Output:
    ```json
    {{{{
    "reasoning_steps": [{reasoning_list}],
    "incorrect_spans" : {incorrect_spans},
    }}}}
    ```
    """
    # Dedent and strip the leading and trailing white spaces
    return dedent(example_template).strip().replace("'","'")



def get_all_reasoning_steps():
    # Regular expression to capture reasoning steps (ensuring multiple steps with proper commas and spaces)
    text = None
    with open("C:\\Users\\karth_2bwktag\\Desktop\\GITHUB\\seminal-mushroom\\labeled_outputs\\id_2042-val\\logging.log","r",encoding="utf-8") as f:
        text = f.read()
    reasoning_steps_pattern = r"---Reasoning Steps---:\s*\[([^\]]+)\]"
    reasoning_steps = re.findall(reasoning_steps_pattern, text)
    return reasoning_steps


import json


def get_soft_probs(soft_labels,text):
    probs = [0]*(len(text)+1)
    for label in soft_labels:
        prob = label.get("prob")
        for i in range(label.get("start"),label.get("end")):
            probs[i] = prob
    return probs

    return 0
def parse_jsonl(jsonl_file):
    # Open and read the JSONL file
    with open(jsonl_file, 'r') as file:
        entries = file.readlines()

    parsed_entries = []
    reasoning_steps = get_all_reasoning_steps()
    few_shot = 10
    with open("p1410_sys.md", "w",encoding="utf-8") as f:
        f.write(get_prompt())
        for count_id,entry in enumerate(entries):
            if(count_id==few_shot):
                break
            # Load the JSON object from the current line
            data = json.loads(entry.strip())
            incorrect_spans = []
            soft_probs = get_soft_probs(data.get("soft_labels"),data.get("model_output_text"))
            # print(soft_probs)
            for hard_label in data.get("hard_labels"):
                start = hard_label[0]
                end = hard_label[1]
                label= {"text":data.get("model_output_text")[start:end],"probability": sum(soft_probs[start:end])/(end-start)}
                label = "{"+str(label).replace("'",'"')+"}"
                incorrect_spans.append(label)
            # Extracting the necessary fields
            context = open("C:\\Users\\karth_2bwktag\\Desktop\\GITHUB\\seminal-mushroom\\data\\context\\en-val.v2_perplexity-sonar-pro\\"+data.get('id')+".context.txt","r",encoding="utf-8").read()
            example = get_example(count_id,context , data.get('model_input', ''), data.get('model_output_text', ''), reasoning_steps[count_id], incorrect_spans)
            # If reasoning steps are available, extract them (if p
            f.write(example)


parse_jsonl("C:\\Users\\karth_2bwktag\\Desktop\\GITHUB\\seminal-mushroom\\data\\val\\mushroom.en-val.v2.jsonl")



