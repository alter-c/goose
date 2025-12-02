import json
import os
import sys
import random
from openai import OpenAI

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

class LLM():
    def __init__(self, problem, domain_pddl, problem_pddl, feat_size=4):
        self.problem = problem
        self.domain_pddl = read_file(domain_pddl)
        self.problem_pddl = read_file(problem_pddl)
        self.feat_size = feat_size

        self.client = self._client()
        self.root_dir = os.environ.get('GOOSE_ROOT')
    
    def _client(self):
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.bianxie.ai/v1"
        )
    
    def _emb_file(self):
        self.emb_dir = os.path.join(self.root_dir, 'experiment', 'embeddings')
        os.makedirs(self.emb_dir, exist_ok=True)
        return os.path.join(self.emb_dir, f"{self.problem.domain_name}.json")
    
    def _exist(self, emb_file):
        if os.path.exists(emb_file):
            emb = load_json(emb_file)
            if len(next(iter(emb.values()))) == self.feat_size:
                return True
            else:
                return False
        else:
            return False

    def create_prompt_task(self):
        PROMPT = "You are an expert in automated planning and semantic representation learning. " + \
                "You will be given a PDDL domain and problem definition. " + \
                "Your task is to generate embeddings for the given planning problem. " + \
                "The embeddings should be numeric vectors with {} dimensions and contain rich semantic information. \n" + \
                "Requirements: " + \
                "1. The embeddings should capture semantic similarity: related elements should be represented by vectors that are closer in the embedding space. But each embedding must be unique, ensuring that no two elements share identical vector values; " + \
                "2. The values of the embeddings should range from -1 to 1; " + \
                "3. The output should be in JSON format, where each entry maps a planning element to its vector; " + \
                "4. Do not generate code, only output the JSON embeddings directly without anything else. \n" + \
                "The example output format is as follows: {} "
        
        TASK = "Domain pddl is \n{} \n" + \
                "Problem pddl is \n{} \n" + \
                "You should output embeddings of below elements: \n" + \
                "predicates: {}, \n" + \
                "actions: {}. \n"

        return PROMPT, TASK

    def create_example_emb(self):
        example_emb = {
            'clear': [random.uniform(-1, 1) for _ in range(self.feat_size)],
            'pick-up': [random.uniform(-1, 1) for _ in range(self.feat_size)]
        }
        return example_emb

    def embedding(self):
        emb_file = self._emb_file()
        if self._exist(emb_file):
            emb = load_json(emb_file)
        else:
            PROMPT, TASK = self.create_prompt_task()
            prompt = PROMPT.format(
                self.feat_size,
                self.create_example_emb()
            )
            task = TASK.format(
                self.domain_pddl,
                self.problem_pddl,
                [pred.name for pred in self.problem.predicates],
                [act.name for act in self.problem.actions]
            )
            
            completion = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": task,
                    }
                ],
                temperature=0.0
            )
            emb_str = completion.choices[0].message.content
            emb = json.loads(emb_str)
            write_json(emb_file, emb)
        
        return emb


if __name__ == "__main__":
    root_dir = os.environ.get('GOOSE_ROOT')
    sys.path.insert(0, os.path.join(root_dir, 'learner'))
    from planning import get_planning_problem

    domain_pddl = os.path.join(root_dir, 'dataset/goose/gripper/domain.pddl')
    problem_pddl = os.path.join(root_dir, 'dataset/goose/gripper/train/gripper-n1.pddl')

    problem = get_planning_problem(domain_pddl, problem_pddl, False)
    llm = LLM(problem, domain_pddl, problem_pddl, feat_size=4)

    emb = llm.embedding()
