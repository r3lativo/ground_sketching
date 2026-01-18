from typing import *
import re
import torch
from datasets import load_dataset, Dataset, load_from_disk
from dataclasses import dataclass, field
import os
import socket
import pandas as pd
from glob import glob
from statistics import mean
from tqdm import tqdm
import base64
import mimetypes
from collections import defaultdict
import gc
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import json
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoProcessor, 
    TrainingArguments, 
    TrainerCallback, 
    TrainerState, 
    TrainerControl, 
    HfArgumentParser
)

import copy
from templates.evaluate_prompts import JUDGE_SYSTEM_PROMPT, get_system_prompt_for_planning, REWARD_ANSWER_SYSTEM_PROMPT_FINAL_ANSWER, REWARD_ANSWER_SYSTEM_PROMPT_PROCESS, QUERY_FORMULATION_SYSTEM_PROMPT

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    judge_name_or_path: Optional[str] = field(
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    image_searcher_model: Optional[str] = field(
        metadata={
            "help": (
                "The model checkpoint for weights initialization of the retriever"
            )
        },
    )
    output_dir: Optional[str] = field(
        metadata={
            "help": (
                "The path to be used to store final evaluations"
            )
        },
    )
    train_dataset_name: Optional[str] = field(
        default="/lustre/fswork/projects/rech/bgp/ucm29gh/code/jeanzay-rl/data/meetup",
        metadata={
            "help": (
                "The path to the dataset to be trained"
            )
        },
    )
    test_dataset_name: Optional[str] = field(
        default="/lustre/fswork/projects/rech/bgp/ucm29gh/code/jeanzay-rl/data/meetup",
        metadata={
            "help": (
                "The path to the dataset to be tested"
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: "},
    )

    relation_type: Optional[str] = field(
        default=None,
        metadata={"help": "What is the type of relation that we are trying to test (eg. spatial, temporal etc.). It is for logging purpose."},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    add_reasoning: bool = field(
        default=False,
        metadata={"help": "Whether to make the extractor reason or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    seed: int = field(
        default=420,
        metadata={"help": "set the seed"},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    server_ip: str = field (
        default="",
        metadata={
            "help": (
                "Server IP Address"
            )
        },
    )

    server_ip_judge: str = field (
        default="",
        metadata={
            "help": (
                "Server IP Address"
            )
        },
    )

    port: str = field (
        default="",
        metadata={
            "help": (
                "Port for the vllm"
            )
        },
    )

    port_judge: str = field (
        default="",
        metadata={
            "help": (
                "Port for the vllm"
            )
        },
    )

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )



    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

class ImageSearcher:
    def __init__(self, model_name='sentence-transformers/clip-ViT-L-14', device='cuda', seed=420):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.current_image_paths = []
        self.current_embeddings = None
        self.seed = seed

    def index_images_from_path(self, path_or_paths):
        """
        Loads images, embeds them, and stores tensors in memory.
        Accepts: Directory String, Single File String, OR List of Strings.
        """
        # Handle List of Paths (The "Zoom-in" logic after the first iterance)
        if isinstance(path_or_paths, list):
            self.current_image_paths = path_or_paths
        
        # Handle Directory
        elif os.path.isdir(path_or_paths): 
            self.current_image_paths = glob(os.path.join(path_or_paths, "*.jpg")) + \
                                       glob(os.path.join(path_or_paths, "*.png"))
        
        # Handle Single File
        else:
            self.current_image_paths = [path_or_paths]

        if not self.current_image_paths:
            print(f"Warning: No images found.")
            self.current_embeddings = None
            return

        # Load and Embed
        try:
            images = [Image.open(p).convert('RGB') for p in self.current_image_paths]
            self.current_embeddings = self.model.encode(images, convert_to_tensor=True, show_progress_bar=False)
        except Exception as e:
            print(f"Error indexing images: {e}")
            self.current_embeddings = None

    def search(self, query: str, k: int = 3):
        if self.current_embeddings is None:
            return []

        query_embedding = self.model.encode([query], convert_to_tensor=True)
        
        # Ensure k isn't larger than the number of available images
        real_k = min(k, len(self.current_image_paths))
        
        hits = util.semantic_search(query_embedding, self.current_embeddings, top_k=real_k)[0]
        
        results = []
        for hit in hits:
            img_idx = hit['corpus_id']
            results.append(self.current_image_paths[img_idx])
            
        return results

class InferenceEvaluator:
    """
    Handles the evaluation of model completions by generating answers and using a judge model to score them.
    """
    def __init__(self, vlm_model, vlm_tokenizer, vlm_processor, vlm_model_name, image_searcher, judge_model, judge_tokenizer, judge_name, seed=420):
        self.vlm_model = vlm_model
        self.vlm_tokenizer = vlm_tokenizer
        self.vlm_processor = vlm_processor
        self.vlm_model_name = vlm_model_name
        self.judge_model = judge_model
        self.judge_tokenizer =judge_tokenizer
        self.judge_name = judge_name
        self.latest_sample_for_logging = None
        self.image_searcher = image_searcher
        self.seed = seed

    def _reasoning_extract_answer(self, input_string: str):
        if '<answer>' in input_string or '</answer>' in input_string or '</reasoning>' in input_string:
            return input_string.split('</reasoning>')[-1].split('<answer>')[-1].split("</answer>")[0].strip()
        else:
            return ''

    def _extract_answer(self, input_string: str):
        return input_string.split('<answer>')[-1].split("</answer>")[0].strip()

    def _encode_image_to_base64(self, image_path):
        """Encodes a local image to a data URL string for the OpenAI API."""
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg" # Default fallback
            
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        return f"data:{mime_type};base64,{encoded_string}"

    def _call_vlm_api(self, text_prompt, image_paths=None, system_prompt="You are a helpful assistant.", temperature: float = 0.7, max_tokens=2048, seed: int = 42):
        """
        Generic wrapper to call the vLLM server via OpenAI API.
        Handles both text-only (Planning) and multimodal (Answering) requests.
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        user_content = []

        # Attach Images (if any)
        if image_paths:
            for p in image_paths:
                base64_image = self._encode_image_to_base64(p)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                })

        # Attach Text
        user_content.append({"type": "text", "text": text_prompt})

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.vlm_model.chat.completions.create(
                model=self.vlm_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self.seed
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling VLM Server: {e}")
            return "Error generating response."
    
    def _judge_with_vllm(self, prompts: List[str], max_output_tokens: int, temperature: float = 0.5, stop: List[str] = None, seed: int = 42):
        if stop is None:
            stop = [self.judge_tokenizer.eos_token]

        response = self.judge_model.completions.create(
            model=self.judge_name,
            prompt=prompts,
            max_tokens=max_output_tokens,
            temperature=temperature,
            stop=stop,
            seed=self.seed
        )
        return [output.text for output in response.choices]
    
    def _get_concrete_query(self, instruction: str, working_memory: str) -> str:
        """Uses the LLM to turn a high-level instruction into a concrete query using context."""
        if not working_memory.strip():
            return instruction

        # concrete_query = self._call_vlm_api(
        #                     text_prompt=f"Context:\n{working_memory}\n\nHigh-Level Instruction:\n{instruction}",
        #                     image_paths=None,
        #                     system_prompt=QUERY_FORMULATION_SYSTEM_PROMPT,
        #                     max_tokens=100
        #                 ).split('assistant\n')[-1].strip()
        concrete_query = self._call_vlm_api(
                            text_prompt=f"Context:\n{working_memory}\n\nHigh-Level Instruction:\n{instruction}",
                            image_paths=None,
                            system_prompt=QUERY_FORMULATION_SYSTEM_PROMPT,
                            max_tokens=100,
                            seed=self.seed
                        ).strip()
        
        return concrete_query

    def _extract_score(self, text: str) -> int:
        # print('score_text', text, flush=True)
        score = text.split("<answer>")[-1]
        score = score.split("</answer>")[0].strip()
        if 'SAME' in score.upper(): 
            return 1
        elif 'DIFFERENT' in score.upper(): 
            return 0
        else:
            print("SCORE NOT IN FORMAT   ", score.upper(), flush=True)
            return 0
    
    def _planned_rag_retrieval(self, image_source_paths, questions: List[str], questioners: List[str], answerers: List[str]):
        # Generate the plans (Text-only call to VLM)
        raw_plans = []
        for i in range(len(questions)):
            plan_prompt = f"\nQuestion from {questioners[i]}: {questions[i]}\nThink and create your plan here:"
            raw_plans.append(
                self._call_vlm_api(
                    text_prompt=plan_prompt,
                    image_paths=None,
                    system_prompt=get_system_prompt_for_planning(answerers[i]),
                    max_tokens=2048
                )
            )

        plans = [self._reasoning_extract_answer(p) for p in raw_plans]

        final_answers = []
        all_retrieved_for_logging = []

        # Execute plan for each question
        for i, plan in enumerate(plans):
            # Each plan has a few steps which need to be broken down
            extracted_plan = [p.strip() for p in plan.split('<item>') if p.strip()]
            print('extracted_plan', extracted_plan, flush=True)
            target_user_folder = "A" if " POV: A" in extracted_plan[0] else "B" if "POV: B" in extracted_plan[0] else answerers[i]
            target_image_source_paths = os.path.join(image_source_paths[i], str(target_user_folder))
            # Start with ALL images in the folder of the target_user
            current_context_images = glob(os.path.join(target_image_source_paths, f'*.png')) + \
                                    glob(os.path.join(target_image_source_paths, "*.jpg"))

            working_memory = ""
            retrieved_steps_for_logging = []

            for step_instruction in extracted_plan:
                try:
                    command, instruction = step_instruction.split(':', 1)
                    command = command.strip().upper()
                    instruction = instruction.strip()
                except:
                    if 'RAG' in step_instruction: 
                        match = re.match(r"RAG\[k=(\d+)\]", step_instruction.strip())
                        if match:
                            command, instruction = match.group(0), step_instruction.strip(match.group(0)).strip()
                        else:
                            command, instruction = 'RAG[k=5]', step_instruction.strip('RAG').strip()
                    elif 'PROCESS' in step_instruction: 
                        command, instruction = 'PROCESS', step_instruction.strip('PROCESS').strip()
                    elif 'FINAL_ANSWER' in step_instruction: 
                        command, instruction = 'FINAL_ANSWER', step_instruction.strip('FINAL_ANSWER').strip()
                    else:
                        continue

                concrete_query = self._get_concrete_query(instruction, working_memory)

                if 'RAG' in command:
                    # Embed the currently available common ground images generated from the dialogues
                    self.image_searcher.index_images_from_path(current_context_images)

                    match = re.match(r"RAG\[k=(\d+)\]", command)
                    if match:
                        k_value = int(match.group(1))
                    else:
                        print('COMMAND NO KEY : ', command)
                        k_value = 3

                    current_context_images = self.image_searcher.search(concrete_query, k=k_value)
                    found_names = [os.path.basename(p) for p in current_context_images]
                    log_msg = f"Found images: {found_names}"
                    working_memory += f"\n--- Result of RAG step: '{concrete_query}' ---\n{log_msg}\n"
                    retrieved_steps_for_logging.append({"step": instruction, "executed_query": concrete_query, "result": log_msg})

                elif command == 'FINAL_ANSWER':
                    processing_prompt = f"Original Question from {questioners[i]} : {questions[i]}.\n\n Final Context:\n{working_memory}\n\nBased on this context, follow this final instruction: {concrete_query}"
                    llm_result = self._call_vlm_api(
                                    text_prompt=processing_prompt,
                                    image_paths=current_context_images,
                                    system_prompt=REWARD_ANSWER_SYSTEM_PROMPT_FINAL_ANSWER
                                )
                    working_memory = llm_result 
                    retrieved_steps_for_logging.append({"step": instruction, "executed_query": concrete_query, "result": llm_result})

                elif command == 'PROCESS': 
                    processing_prompt = f"Current Context:\n{working_memory}\n\nInstruction: {concrete_query}"                    
                    llm_result = self._call_vlm_api(
                                    text_prompt=processing_prompt,
                                    image_paths=current_context_images,
                                    system_prompt=REWARD_ANSWER_SYSTEM_PROMPT_PROCESS
                                )

                    working_memory = f"--- Result of PROCESS step: '{concrete_query}' ---\n{llm_result}\n"
                    retrieved_steps_for_logging.append({"step": instruction, "executed_query": concrete_query, "result": llm_result})

            final_answers.append(working_memory) # The final state of the memory is the answer
            all_retrieved_for_logging.append(retrieved_steps_for_logging)
                
        return final_answers, all_retrieved_for_logging, plans

    def evaluate(self, questions: List[str], questioners: List[str], answerers: List[str], correct_answers: List[str], image_path: str, datapoint_id: str):
        """
        Performs the full inference and evaluation pipeline for a single data point.
        
        Returns:
            float: The average score for the given data point.
        """
        llm_answers, retrieved_steps, plans = self._planned_rag_retrieval(
            questions=questions,
            questioners=questioners,
            answerers=answerers,
            image_source_paths=image_path
        )

        llm_answers = [self._reasoning_extract_answer(l) for l in llm_answers]
        # llm_answers, retrieved_steps, plans = self._retrieval(common_ground, questions, questioners)

        # Use the model as a judge to score the generated answers
        judgement_prompts = []
        for i, llm_response in enumerate(llm_answers):
            # Extract only the generated part of the response
            judge_user_prompt = f"Question: {questions[i]}\n\nCorrect Response: {correct_answers[i]}\n\nLLM Response: {llm_response}\n\nNow give your judgement keeping in mind the format."
            input_message = [{'role': 'system', 'content': JUDGE_SYSTEM_PROMPT}, {'role': 'user', 'content': judge_user_prompt}]
            formatted_judge_prompt = self.judge_tokenizer.apply_chat_template(input_message, tokenize=False, add_generation_prompt=True)
            judgement_prompts.append(formatted_judge_prompt)
        
        decoded_judge_outputs = self._judge_with_vllm(
                                prompts=judgement_prompts,
                                max_output_tokens=200,
                                temperature=0.5
                            )

        # Extract scores and log the first sample for inspection
        scores = []
        for i, judge_output_text in enumerate(decoded_judge_outputs):
            judge_response_content = judge_output_text.split('assistant\n')[-1]
            score = self._extract_score(judge_response_content.split('</reasoning>')[-1])
            scores.append(score)

            # Store the details of the first evaluated sample for logging
            if i == 0:
                self.latest_sample_for_logging = {
                    "question": questions[i],
                    "plan": plans[i],
                    "retrieved_steps": retrieved_steps[i],
                    "llm_response": llm_answers[i],
                    "judge_response": judge_response_content,
                    "correct_response": correct_answers[i],
                    "score": score,
                    "datapoint_id": datapoint_id
                }

        gc.collect()
        return mean(scores) if scores else 0.0

def load_meetup_data(meetup_root: str) -> Dataset:
    """Load and format Meetup dataset as HF Dataset compatible with GRPOTrainer."""
    data_records = []

    # folders = sorted(glob(os.path.join(meetup_root, '*')))
    # for folder in folders:
    for csv_path in glob(os.path.join(meetup_root, '**', '*.csv'), recursive=True, include_hidden=False):
        try:
            df = pd.read_csv(csv_path)
    
            if df.shape[0] < 3:
                continue  # not enough messages for context, question, answer

            # Build conversation log from all but last 2 rows
            df_context = df.iloc[:-3]
            conversation_log = "Here is the conversation -\n"
            for _, row in df_context.iterrows():
                conversation_log += f"Speaker: {row['character']}\nTime: {row['time']}\nMessage: {row['text']}\n\n"

            question_row = df.iloc[-3]
            question = question_row['text']
            question_user = question_row['character']
            answer_user = df.iloc[-2]['character']
            model_answer = df.iloc[-2]['text']

            # Get the path where the common ground images generated for this conversation exist
            base_dir = os.path.dirname(csv_path)
            image_path = os.path.join(base_dir, 'images')
            base_name = os.path.basename(csv_path)
            file_name = os.path.splitext(base_name)[0]

            data_records.append({
                'question': [question],
                'questioner': [question_user],
                'answerer': [answer_user],
                'model_answer': [model_answer],
                'image_path': [image_path],
                'file_name': [file_name]
            })

        except Exception as e:
            print(f"Skipping file {csv_path} due to error: {e}")

    return Dataset.from_list(data_records) 

def setup_eval_logger(output_dir: str, filename: str = "eval_samples.log") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)

    logger = logging.getLogger("eval_samples")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate logs if root logger is configured elsewhere

    # Avoid adding handlers multiple times if main_infer is called more than once
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    # stash the path for convenience
    logger.log_path = log_path  # type: ignore[attr-defined]
    return logger


def main_infer(model_args):
    eval_logger = setup_eval_logger(model_args.output_dir, f"{model_args.relation_type}.log")
    print(f"[Logging] Writing evaluation samples to: {eval_logger.log_path}", flush=True)
    
    print("Setting up local embedding model for retriever...")
    image_searcher = ImageSearcher(model_name=model_args.image_searcher_model, seed=model_args.seed)
    try:
        print("Testing the image searcher model...")
        test_vec = image_searcher.model.encode(["query: This is a test."], convert_to_tensor=True)
        print(f"Success! Vector starts with: {test_vec[:5]}")
    except Exception as e:
        print(f"FAILED: The image searcher model could not be used. Error: {e}")
        import traceback
        traceback.print_exc()
        return
    print("Image searcher model setup complete.")

    print(f"Connecting to vLLM server at http://{model_args.server_ip}:{model_args.port}/v1")
    vlm_model = OpenAI(
        base_url=f"http://{model_args.server_ip}:{model_args.port}/v1",
        api_key="not-needed"
    )

    vlm_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    vlm_tokenizer.pad_token = vlm_tokenizer.eos_token
    vlm_tokenizer.padding_side = "left"

    vlm_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    judge_model = OpenAI(
        base_url=f"http://{model_args.server_ip}:{model_args.port_judge}/v1",
        api_key="not-needed"
    )

    judge_tokenizer = AutoTokenizer.from_pretrained(model_args.judge_name_or_path)
    judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_tokenizer.padding_side = "left"

    eval_dataset = load_meetup_data(model_args.test_dataset_name)

    evaluator = InferenceEvaluator(vlm_model, vlm_tokenizer, vlm_processor, model_args.model_name_or_path, image_searcher, judge_model, judge_tokenizer, model_args.judge_name_or_path, seed=model_args.seed)

    total_scores = []
    
    for i, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        with torch.no_grad():
            score = evaluator.evaluate(
                questions=sample['question'],
                questioners=sample['questioner'],
                answerers=sample['answerer'],
                correct_answers=sample['model_answer'],
                image_path=sample['image_path'],
                datapoint_id=sample['file_name']
            )
        total_scores.append(score)

        # Print a detailed sample output every N steps
        if i % 1 == 0:
            log_sample = evaluator.latest_sample_for_logging
            if log_sample:
                block = (
                        f"\n--- Logging Evaluation Sample at Step {i+1} for ID {log_sample['datapoint_id']} ---\n"
                        f"\nQuestion:\n{log_sample['question']}\n"
                        f"\nPlan:\n{log_sample['plan']}\n"
                        f"\nRetrieved Steps:\n{log_sample['retrieved_steps']}\n"
                        f"\n[LLM Response]:\n{log_sample['llm_response']}\n"
                        f"\n[Correct Response]:\n{log_sample['correct_response']}\n"
                        f"\n[Full Output from Judge Model]:\n{log_sample['judge_response']}\n"
                        f"\n[Extracted Score]: {log_sample['score']}\n"
                        f"--------------------------------------------------\n"
                    )
                
                print(block, flush=True)
                eval_logger.info(block)


    # Calculate and Print Final Accuracy
    average_accuracy = mean(total_scores) if total_scores else 0.0
    avg_blk = (
        f"\n============= Final Results ============="
        f"Total samples evaluated: {len(total_scores)}"
        f"Average Accuracy Score: {average_accuracy:.4f}"
        f"======================================="
    )

    print(avg_blk, flush=True)
    eval_logger.info(avg_blk)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args_into_dataclasses()[0]
    
    main_infer(model_args)

 