#!/usr/bin/env python
# coding: utf-8

import abc
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import *

from beartype import beartype
import datasets
import fire
import numpy as np
import more_itertools
import queue
import rich
import torch
import torch.nn as nn
import transformers
import wandb

try:
    import colored_traceback.auto
except ImportError:
    pass


TokenizerType = Union[
    transformers.tokenization_utils_fast.PreTrainedTokenizerFast, 
    transformers.PreTrainedTokenizer
]


###############################################################################
# Epsilon Scheduling
###############################################################################
class BaseEpsilonScheduler(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        pass

class LinearEpsilonScheduler(BaseEpsilonScheduler):
    def __init__(self, epsilon, num_steps):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.epoch = 0

    def __call__(self):
        self.epoch += 1
        epsilon = min(self.epsilon * (1 - self.epoch / self.num_epochs), 1)
        wandb.log({"epsilon": epsilon})
        wandb.log({"epsilon_num_steps": self.num_steps})
        return epsilon

class ConstantEpsilonScheduler(BaseEpsilonScheduler):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self):
        epsilon = self.epsilon
        wandb.log({"epsilon": epsilon})
        return epsilon

###############################################################################
# Retrievers
###############################################################################
class BaseRetriever(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, query_ids, query_index):
        pass

class StupidRetriever(BaseRetriever): 
    @beartype
    def __init__(
        self, 
        *, 
        model: torch.nn.Module, 
        tokenizer: TokenizerType, 
        device: Union[int, str], 
        train_vectors: torch.Tensor, 
        train_samples_dict: Dict[str, Any],
    ):
    
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.train_vectors = train_vectors
        self.train_samples_dict = train_samples_dict

    def retrieve(self, query_index):
        # Get the representation
        representation = self.train_vectors[query_index]
        with torch.inference_mode():
            # Compute the inner products
            scores = torch.matmul(representation, self.train_vectors.t())
            # Get the top 2 results, to potentially exclude the sample itself.
            topk = torch.topk(scores, k=2, dim=-1)
        topk = topk.indices.cpu().numpy()
        
        for retrieved_idx in topk:
            if retrieved_idx != query_index:
                return {k: v[retrieved_idx] for k, v in self.train_samples_dict.items()} | {"index": retrieved_idx}
        
# build train vectors
@beartype
def make_retrival_model_and_vectors(
    *,
    retriever_name: str, 
    path_to_vectors: Union[str, Path], 
    device: Union[int, str], 
) -> BaseRetriever:
    """We expect the dir to have the following structure:
    - config.json
    - train_samples.json 
    - train_vectors.npy
    """    
    # Make some checks
    retriever_model = transformers.AutoModel.from_pretrained(retriever_name)
    retriever_tokenizer = transformers.AutoTokenizer.from_pretrained(retriever_name)

    with open(path_to_vectors / "train_samples.json") as f:
        train_samples_dict = json.load(f)
        

    vectors = torch.tensor(np.load(path_to_vectors / "train_vectors.npy")).to(device)
    retriever = StupidRetriever(
        model=retriever_model, 
        tokenizer=retriever_tokenizer, 
        device=device, 
        train_vectors=vectors, 
        train_samples_dict=train_samples_dict,
    )
    
    return retriever


###############################################################################
# Iterator and Priority Queue stuff
###############################################################################
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class BoostingIterator(torch.utils.data.IterableDataset):
    @beartype
    def __init__(
        self, 
        *, 
        dataset, 
        retriever_client: BaseRetriever, 
        classifier: nn.Module, seed: int, 
        classification_device: Union[int, str], 
        classification_tokenizer: TokenizerType, 
        retriever_device: Union[int, str],
        epsilon_scheduler: BaseEpsilonScheduler, 
        loss_ema_alpha: float, 
        score_mode: str,
        score_mode_config: Dict[str, Any],
        config: Dict[str, Any],
    ):
        super().__init__()
        self.dataset = dataset.map(
            lambda _, idx:{"index": idx}, with_indices=True, 
        ).shuffle(seed=seed)
        self.dataset = self.dataset.remove_columns(["idx"])
        self.priority_queue = queue.PriorityQueue()
        self.retriever_client = retriever_client
        self.epsilon_scheduler = epsilon_scheduler
        self.randomizer = np.random.RandomState(seed)
        self.seed = seed
        self.dataset_iter = None
        self.classifier = classifier
        self.classification_tokenizer = classification_tokenizer
        self.classification_device = classification_device
        self.retriever_device = retriever_device
        self.variance_rolling_average = None
        self.loss_ema_alpha = loss_ema_alpha
        self.dataset_type = config["dataset_type"]
        self.seen_samples = 0
        self.epoch_length = len(self.dataset)
        self.total_num_steps = 0
        self.score_mode = score_mode
        self.score_mode_config = score_mode_config
        self.loss_moving_average = None

        if self.score_mode == "fixed_loss" or self.score_mode == "step_sensitive_fixed_loss":
            assert self.fixed_loss is not None


        if self.dataset_type == "dual_entry_classification":
            self.field_a_name = config["field_a_name"]
            self.field_b_name = config["field_b_name"]

        assert "idx" not in self.dataset

        # assert mode in ["epsilon_priority_no_reset", "pure_sampled", "epsilon_sampled"], mode

    def push_score(self, inputs, loss):
        loss: Final = loss
        inputs: Final = inputs

        with torch.inference_mode():
            ################################################################################
            # Moving average of the loss.
            ################################################################################    
            uniform_samples_loss: Final = loss[torch.logical_not(inputs["is_retrieved"])].detach()

            # Only use the uniform random samples to evaluate the batch's average loss.
            # Protection against the edge case where everything is retrieved.
            if len(uniform_samples_loss) != 0:
                average_loss: Final = torch.mean(uniform_samples_loss).detach().cpu().numpy()
                if self.loss_moving_average is None:
                    self.loss_moving_average = average_loss
                else:
                    self.loss_moving_average = (
                        self.loss_ema_alpha * self.loss_moving_average + (1 - self.loss_ema_alpha) * average_loss
                    )
                wandb.log({"loss_moving_average": self.loss_moving_average})

            ################################################################################
            # Scores the inputs and pushes them to the priority queue.
            ################################################################################

            for i, (input_, mask, loss_, index) in (
                enumerate(more_itertools.zip_equal(inputs["input_ids"], inputs["attention_mask"], loss, inputs["index"]))
            ):
                
                loss_ = loss_.detach().cpu().numpy()
                relative_loss = loss_ / self.loss_moving_average
                

                if inputs["has_previous_loss"][i]:
                    previous_loss = inputs["previous_loss"][i]

                    wandb.log({"previous_relative_loss": previous_loss})
                    wandb.log({"current_relative_loss": relative_loss})
                    wandb.log({"ratio_relative_losses": relative_loss / previous_loss})
                    wandb.log({"average_relative_losses_check": (relative_loss + previous_loss) / 2})

                assert loss_.shape == torch.Size([]), loss_.shape

                if self.score_mode == "relative_loss":
                    score = - relative_loss
                elif self.score_mode == "step_sensitive_relative_loss":
                    score = - relative_loss * self.total_num_steps
                elif self.score_mode == "fixed_loss":
                    score = np.abs(loss_ - self.fixed_loss)
                elif self.score_mode == "step_sensitive_fixed_loss":
                    score = np.abs(loss_ - self.fixed_loss) * self.total_num_steps 
                else:
                    raise ValueError(f"Unknown score mode: {self.score_mode}")

                self.priority_queue.put(
                    PrioritizedItem(
                            priority=score, 
                            item=dict(input_ids=input_, attention_mask=mask, index=index, previous_loss=relative_loss)
                        )
                    )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.dataset_iter is None:
            self.dataset_iter = iter(self.dataset)
        self.seen_samples = 0
        return self
    
    def __next__(self):
        if self.seen_samples == self.epoch_length:
            raise StopIteration

        # This next is only called by the training dataset.
        self.total_num_steps += 1

        # Test if we have a sample and if we pass the epsilon threshold
        empty = self.priority_queue.empty()
        rand = self.randomizer.rand()
        if not empty and rand < self.epsilon_scheduler():
            ################################################################################
            # Retrieved sample
            ################################################################################
            # pull a sample from the priority queue
            sample = self.priority_queue.get().item
            
            # We retrieve the next sample.
            next_sample = self.retriever_client.retrieve(sample["index"])
            
            
            next_sample["is_retrieved"] = True
            next_sample["previous_loss"] = sample["previous_loss"]
            next_sample["has_previous_loss"] = True
        else:
            ################################################################################
            # Uniform random sample
            ################################################################################
            try:
                next_sample = next(self.dataset_iter)  
            except StopIteration:
                self.dataset = self.dataset.shuffle(seed=self.seed)
                self.dataset_iter = iter(self.dataset)
                next_sample = next(self.dataset_iter)  
            next_sample["is_retrieved"] = False
            next_sample["previous_loss"] = float('nan')
            next_sample["has_previous_loss"] = False

        ################################################################################
        # Per dataset type preparation
        ################################################################################
        if self.dataset_type == "single_entry_classification":
            tokenized = self.classification_tokenizer.encode_plus(
                next_sample["inputs"], 
                truncation=True, 
                padding=True,    
            )
            del next_sample["inputs"]
        elif self.dataset_type == "dual_entry_classification":
            tokenized = self.classification_tokenizer.encode_plus(
                next_sample[self.field_a_name], 
                next_sample[self.field_b_name], 
                truncation=True, 
                padding=True,
            )
            del next_sample[self.field_a_name]
            del next_sample[self.field_b_name]
            
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        # import html
        # if next_sample["is_retrieved"]:
        #     wandb.log(
        #         {
        #         "thing_pair":[
        #             wandb.Html(f"<b>previous:</b>      {sample['index']}      {html.escape(self.classification_tokenizer.decode(sample['input_ids']   ))})"),
        #             wandb.Html(f"<b>current:&nbsp;</b> {next_sample['index']} {html.escape(self.classification_tokenizer.decode(tokenized['input_ids']))})")
        #         ]
        #         }
        #        
        #     )
        # rich.print(
        #     f"[bold green]previous:[/]  {sample['index']} {self.classification_tokenizer.decode(sample['input_ids'])}\n", 
        #     f"[bold green]current:[/]   {next_sample['index']} {self.classification_tokenizer.decode(tokenized['input_ids'])}"
        # )
        # print("#" * 80)

        assert len(tokenized.keys() & next_sample.keys()) == 0, (tokenized.keys(), next_sample.keys()) 
        retval = dict(**tokenized, **next_sample)
        assert "previous_loss" in retval, retval.keys()
        return retval



###############################################################################
# Custom trainer stuff
###############################################################################
class BoostingTrainer(transformers.Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        assert "labels" in inputs, inputs.keys()
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        assert labels is None


        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            assert False
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. 
                Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        
        ################################################################################
        # Remove the parts of the inputs that model.forward does not need.
        ################################################################################
        inputs = self._prepare_inputs(inputs)
        index = inputs["index"]
        is_retrieved = inputs["is_retrieved"]
        previous_loss = inputs["previous_loss"]
        has_previous_loss = inputs["has_previous_loss"]
        del inputs["previous_loss"]
        del inputs["is_retrieved"]
        del inputs["index"]
        del inputs["has_previous_loss"]

        with self.autocast_smart_context_manager():
            # Get the loss
            loss, outputs = self.compute_loss(model=model, inputs=inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            # Mean over per gpu averages
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # This is ignored in the priority queue computation
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            assert False
            # Deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            assert False
            with torch.cuda.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            assert False
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        loss = loss.detach()

        # Addition for RetroBoost
        # Make sure the losses are similar, then push them to the priority queue
        # Put index back in

        inputs["index"] = index
        inputs["is_retrieved"] = is_retrieved
        inputs["previous_loss"] = previous_loss
        inputs["has_previous_loss"] = has_previous_loss

        with torch.inference_mode():
            loss_per_sample = torch.nn.functional.cross_entropy(outputs.logits.detach(), inputs["labels"].detach(), reduction="none")
            assert loss_per_sample.ndim == 1, loss_per_sample.ndim
            loss_per_gpu = torch.mean(loss_per_sample, dim=0)
            computed_loss = torch.mean(loss_per_gpu)
            # rich.print("[red bold]logits[/]", outputs.logits.detach().cpu().numpy())
            # rich.print("[red bold]logits[/]", outputs.logits.detach().cpu().numpy().shape)
            # rich.print("[red bold]LOSS[/]", loss.detach().cpu().numpy(), " [red bold]computed_loss[/]", computed_loss)
            # assert torch.allclose(loss, computed_loss)

            self.get_train_dataloader().dataset.push_score(inputs, loss_per_sample)

        return loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return datasets.load_metric("accuracy").compute(predictions=predictions, references=labels)


@beartype
@dataclass
class DataCollatorWithPadding:
    tokenizer: transformers.data.data_collator.PreTrainedTokenizerBase
    padding: Union[bool, str, transformers.data.data_collator.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # check that they all have the same keys
        all_keys = set()
        for feature in features:
            all_keys |= feature.keys()
        
        for feature in features:
            assert all_keys == feature.keys(), all_keys - feature.keys()

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        return batch

@beartype
@dataclass
class RunConfig:
    """ Allows to make sure everything expected is in the json config and of the right type.
    """
    run_name: str
    regular_trainer: bool

    dataset_tuple: List[str]
    classifier_name: str

    classifier_batch_size: int
    epsilon_scheduler_type: str
    epsilon_scheduler_config: Dict[str, Any]
    loss_ema_alpha: float
    
    score_mode: str
    score_mode_config: Dict[str, Any]


def main(config_path: str, script_dir: str = Path(__file__).absolute().parent):
    # Short name mode: if we just get "us_large" instead of "configs/us_large.json", 
    # we are in short name mode, we load use it to load the config. 
    # We don't try to complete "us_large.json" because it might exist, just "us_large".
    # We feel it's a good compromise.
    # if Path(config_path).name == config_path and not Path(config_path).suffix:
    #     config_path = SCRIPT_DIR / "run_configs" / config_path
    #     config_path = config_path.parent / (config_path.name + ".json")

    ###############################################################################
    # Load the config
    ###############################################################################
    with Path(config_path).open("r") as f:
        meta_param_config = RunConfig(**json.load(f))

    # Things that don't change
    WEIGHT_DECAY = 0.01
    LEARNING_RATE = 1e-5
    ENABLE_FP16 = True
    RETRIEVER_NAME = "facebook/contriever"
    PATH_TO_VECTORS = SCRIPT_DIR / f"./vectors_{'_'.join(meta_param_config.dataset_tuple)}_{RETRIEVER_NAME.split('/')[-1]}/"
    CLASSIFIER_EVAL_BATCH_SIZE_MULTIPLIER = 1.5
    CLASSIFIER_DEVICE = "cuda"
    RETRIEVER_DEVICE = "cuda"
    SEED = 0
    SPLIT_RATIO = 0.85
    NUM_EPOCHS_TO_TRAIN_ON = 60

    ###############################################################################
    # Fast setup 
    ###############################################################################
    dataset_config: Final = json.loads((PATH_TO_VECTORS / "config.json").read_text())
    assert dataset_config["retriever_name"] == RETRIEVER_NAME, (
        f"{dataset_config['retriever_name'] != RETRIEVER_NAME}"
    )

    wandb_config = dict(
        classifier_batch_size=meta_param_config.classifier_batch_size,
        classifier_name=meta_param_config.classifier_name,
        dataset_tuple=meta_param_config.dataset_tuple,
        epsilon=dict(
            scheduler_type=meta_param_config.epsilon_scheduler_type,
            scheduler_config=meta_param_config.epsilon_scheduler_config,
        ),
        loss_ema_alpha=meta_param_config.loss_ema_alpha,
        random_seed=SEED,
        regular_trainer=meta_param_config.regular_trainer,
        retriever_name=RETRIEVER_NAME,
        split_ratio=SPLIT_RATIO,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        dataset_type=dataset_config["dataset_type"],
        enable_fp16=ENABLE_FP16,
    )

    wandb.require(experiment="service")
    wandb.init(
        config=wandb_config,
        project="RetroBoost", 
        entity="retroboost",
        name=meta_param_config.run_name,
    )

    EPSILON_SCHEDULER_MAP = dict(
        constant=ConstantEpsilonScheduler,
    )

    # Random seeds. 
    np.random.seed(0)
    torch.manual_seed(0)

    classifier_tokenizer: Final = transformers.AutoTokenizer.from_pretrained(
        meta_param_config.classifier_name
    )

    # Load the config

    # Load the datasets
    dataset_train: Final = datasets.load_dataset(
        *meta_param_config.dataset_tuple, split=f"train"
    )
    dataset_validation: Final = datasets.load_dataset(
        *meta_param_config.dataset_tuple, split=f"validation"
    )

    ALL_LABELS = set(dataset_train["label"])
    NUM_LABELS = len(ALL_LABELS)
    assert ALL_LABELS == set(range(NUM_LABELS))

    # Delete the extra fields
    if dataset_config["dataset_type"] == "dual_entry_classification":
        fields = dataset_train[0].keys()
        dataset_train.remove_columns(
            fields - {dataset_config["field_a_name"], dataset_config["field_b_name"], "label"}
        )

    def preprocess_function(examples, tokenizer, config):
        if dataset_config["dataset_type"] == "single_entry_classification":
            return tokenizer(examples["text"], truncation=True, padding=True)
        elif dataset_config["dataset_type"] == "dual_entry_classification":
            return tokenizer(
                examples[dataset_config["field_a_name"]], 
                examples[dataset_config["field_b_name"]], 
                truncation=True, 
                padding=True,
            )

        raise ValueError(f"Unknown dataset type {dataset_config['dataset_type']}")

    tokenized_training: Final = dataset_train.map(
        lambda examples: preprocess_function(examples, classifier_tokenizer, dataset_config), 
        batched=True
    ).shuffle(seed=SEED)

    tokenized_validation: Final = dataset_validation.map(
        lambda examples: preprocess_function(examples, classifier_tokenizer, dataset_config), 
        batched=True
    ).shuffle(seed=SEED)

    training_args: Final = transformers.TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=20,
        output_dir="./results",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=meta_param_config.classifier_batch_size,
        per_device_eval_batch_size=int(
            meta_param_config.classifier_batch_size * CLASSIFIER_EVAL_BATCH_SIZE_MULTIPLIER
        ),
        num_train_epochs=NUM_EPOCHS_TO_TRAIN_ON,
        report_to="wandb",
        weight_decay=WEIGHT_DECAY,
        fp16=ENABLE_FP16,
    )

    retriever: Final = make_retrival_model_and_vectors(
        retriever_name=RETRIEVER_NAME, 
        path_to_vectors=PATH_TO_VECTORS, 
        device=RETRIEVER_DEVICE, 
    )
    retriever_client: Final = retriever

    classifier: Final = transformers.AutoModelForSequenceClassification.from_pretrained(
        meta_param_config.classifier_name, num_labels=NUM_LABELS
    )

    classifier.config.problem_type = "single_label_classification"

    if meta_param_config.regular_trainer:
        TrainerClass = transformers.Trainer    
        ds_train = tokenized_training
    else:
        TrainerClass = BoostingTrainer
        
        ds_train = BoostingIterator(
            dataset=dataset_train, 
            retriever_client=retriever_client, 
            classifier=classifier, 
            epsilon_scheduler=EPSILON_SCHEDULER_MAP[
		meta_param_config.epsilon_scheduler_type](
                **meta_param_config.epsilon_scheduler_config), 
            seed=SEED,
            retriever_device=RETRIEVER_DEVICE, 
            classification_device=CLASSIFIER_DEVICE,
            classification_tokenizer=classifier_tokenizer,
            loss_ema_alpha=meta_param_config.loss_ema_alpha,
            score_mode=meta_param_config.score_mode,
            score_mode_config=meta_param_config.score_mode_config,
            config=meta_param_config,
        )

    trainer = TrainerClass(
            model=classifier,
            args=training_args, 
            tokenizer=classifier_tokenizer, 
            train_dataset=ds_train, 
            eval_dataset=tokenized_validation,
            data_collator=transformers.data.data_collator.DataCollatorWithPadding(
                tokenizer=classifier_tokenizer
            ),
            compute_metrics=compute_metrics,
        )

    output = trainer.train()
    print(output)
    print(trainer.evaluate())


if __name__ == "__main__":
    fire.Fire(main)
