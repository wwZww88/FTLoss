import os

import sys
sys.path.append("/FTLoss/src")

import random
random.seed(88)

import re
from collections import defaultdict

from datasets import Dataset

"""
import KnowledgeGraph
import importlib
importlib.reload(KnowledgeGraph)
"""

from KnowledgeGraph import GraphMerger
merger = GraphMerger()

def find_local_dataset_path(dataset_name):
    """
    Find dataset path from local cache.
    
    Args:
        dataset_name (str): Hugging Face dataset identifier (e.g., "PKU-Alignment/BeaverTails")
    
    Returns:
        str: Path to the cached dataset snapshot, or None if not found
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    dataset_name_safe = dataset_name.replace("/", "--")
    dataset_cache = os.path.join(cache_dir, f"datasets--{dataset_name_safe}")
    
    if not os.path.exists(dataset_cache):
        print(f"Dataset cache not found at {dataset_cache}")
        return None
    
    # Look for snapshots directory
    snapshots_dir = os.path.join(dataset_cache, "snapshots")
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            # Get the latest snapshot
            latest_snapshot = sorted(snapshots)[-1]
            dataset_path = os.path.join(snapshots_dir, latest_snapshot)
            print(f"Found dataset at: {dataset_path}")
            return dataset_path
    
    return None

def load_selected_concepts():
    """
    Load sample information for selected concepts.
    Return:
        concept_ids: dict {'concept': [id1, id2, ...]}
        all_ids: list [id1, id2, ...]
    """
    
    with open("/FTLoss/concept_selection_keep_result.json", 'r') as f:
        keep = eval(f.read())
    
    concept_ids = {}
    all_ids = []
    for item in keep:
        concept = item['concept']
        source_sample_ids = merger.graph.nodes[concept]['source_sample_ids']
        
        concept_ids[concept] = source_sample_ids
        all_ids.extend(source_sample_ids)
    
    return concept_ids, list(set(all_ids))

def split_dataset(concept_ids, test_size=5):
    """
    Split train dataset and test dataset for selected concepts. 
    This function construct the test dataset with balanced safe/unsafe samples for each concept.
    Select test, then rest for train.
    """
    train_dataset, test_dataset = {}, {}
    
    both_enough = 0
    only_safe_enough = 0
    only_unsafe_enough = 0
    both_unenough = 0
    
    for c, ids in concept_ids.items():
        safe_list = [id for id in ids if merger.sample_dict[id]['is_safe']]
        unsafe_list = [id for id in ids if not merger.sample_dict[id]['is_safe']]
        
        random.shuffle(safe_list)
        random.shuffle(unsafe_list)
        
        safe_enough   = len(safe_list)   >= test_size
        unsafe_enough = len(unsafe_list) >= test_size
        
        if safe_enough and unsafe_enough:
            train_dataset[c] = {
                'safe': safe_list[test_size:],
                'unsafe': unsafe_list[test_size:]
            }
            test_dataset[c] = {
                'safe': safe_list[: test_size],
                'unsafe': unsafe_list[: test_size]
            }
            both_enough += 1
            
        elif safe_enough and not unsafe_enough:
            train_dataset[c] = {
                'safe': safe_list[test_size:],
                'unsafe': []
            }
            test_dataset[c] = {
                'safe': safe_list[: test_size],
                'unsafe': unsafe_list
            }
            only_safe_enough += 1
            
        elif not safe_enough and unsafe_enough:
            train_dataset[c] = {
                'safe': [],
                'unsafe': unsafe_list[test_size:]
            }
            test_dataset[c] = {
                'safe': safe_list,
                'unsafe': unsafe_list[: test_size]
            }
            only_unsafe_enough += 1
            
        else: # not enough for both
            train_dataset[c] = {
                'safe': [],
                'unsafe': []
            }
            test_dataset[c] = {
                'safe': safe_list,
                'unsafe': unsafe_list
            }
            both_unenough += 1
            
    # print(both_enough, only_safe_enough, only_unsafe_enough, both_unenough)
    return train_dataset, test_dataset

def augmentation(concept_list):
    """
    Find sample to related concept from failed graph extraction files.
    """

    # Pre-compilation: Merge all concepts into one regular expression to scan the text only once
    # \b ensures word boundaries to prevent 'lock' from matching 'lockdown'
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(c) for c in concept_list) + r')\b',
        re.IGNORECASE
    )
    map_to_original = {c.lower(): c for c in concept_list}
    
    def find_concepts_in_sample(text, pattern=pattern):
        """Find concepts in text."""
        matches = pattern.findall(text)
        
        # Deduplicate then convert matched pattern to its original form in concept_list
        matches = list({map_to_original[m.lower()] for m in matches})
        return matches
    
    files = [
        "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_failed_0-10000.json",
        "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_failed_10000-20000.json",
        "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_failed_20000-27185.json",
        "/FTLoss/results/knowledge/beavertails/beavertails_30k_test_data_knowledge_failed_0-3021.json",
        
        "/FTLoss/results/knowledge/beavertails/beavertails_330k_train_data_knowledge.json",
        "/FTLoss/results/knowledge/beavertails/beavertails_330k_test_data_knowledge.json"
        ]
    
    augment = {c:{'safe': [], 'unsafe': []} for c in concept_list}

    for file in files:
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            
        # print(f"{files[0].split('/')[-1]}: {len(lines)}")
        
        for line in lines[1:]:
            try:
                info = eval(line.split('"""')[1])
                text = info['prompt'] + ' ' + info['response']
                matches = find_concepts_in_sample(text)

                # sample_id =  f"test-{info['id']}" if "_test_" in file else f"train-{info['id']}"
                
                if   "_30k_train" in file:
                    sample_id = f"train30k-{info['id']}"
                elif "_30k_test" in file:
                    sample_id = f"test30k-{info['id']}"
                elif "_330k_train" in file:
                    sample_id = f"train330k-{info['id']}"
                elif "_330k_test" in file:
                    sample_id = f"test330k-{info['id']}"
                        
                is_safe = 'safe' if info['is_safe'] else 'unsafe'
                for c in matches:
                    augment[c][is_safe].append(sample_id)
            except:
                continue
    
    return augment

def combine_dataset(
    train_dataset, test_dataset, augment_dataset,
    test_size = 10
    ):
    """
    Combine test_dataset and train_dataset with the augment_dataset.
    If the augmented test dataset includes concepts that does not meet the text_size requirement, we abandon.
    
    """
    # test_size = max([max([len(split['safe']), len(split['unsafe'])]) for split in test_dataset.values()])

    combine_train_dataset, combine_test_dataset = {}, {}
    for c, split in test_dataset.items():
        
        test_safe_need = test_size - len(split['safe'])
        test_unsafe_need = test_size - len(split['unsafe'])
        # print(test_safe_need, test_unsafe_need)
        
        # Fill test dataset to test_size
        combine_test_dataset[c] = {
            'safe': test_dataset[c]['safe'] + augment_dataset[c]['safe'][:test_safe_need],
            'unsafe': test_dataset[c]['unsafe'] + augment_dataset[c]['unsafe'][:test_unsafe_need],
        }
        
        # Fill train dataset using the rest of augment
        combine_train_dataset[c] = {
            'safe': train_dataset[c]['safe']+ augment_dataset[c]['safe'][test_safe_need:],
            'unsafe': train_dataset[c]['unsafe']+ augment_dataset[c]['unsafe'][test_unsafe_need:],
        }
    
    """
    combine_train_dataset = {c: {
            'safe': train_dataset[c]['safe']+ augment_dataset[c]['safe'],
            'unsafe': train_dataset[c]['unsafe']+ augment_dataset[c]['unsafe'],
        } for c in train_dataset.keys()}
    """
    
    # print(combine_test_dataset['weed'])
    
    # Abandon concepts that has < test_size samples
    concepts_to_abandon = [c for c, split in combine_test_dataset.items() 
                    if len(split['safe']) < test_size or len(split['unsafe']) < test_size]
    
    combine_test_dataset_abandon = {c: split for c, split in combine_test_dataset.items() 
                            if c not in concepts_to_abandon}
    combine_train_dataset_abandon = {c: split for c, split in combine_train_dataset.items() 
                            if c not in concepts_to_abandon}
    
    return combine_train_dataset_abandon, combine_test_dataset_abandon
    # return combine_train_dataset, combine_test_dataset
    
def construct_outdomain(all_ids):
    """
    Construct OutDomain dataset, samples that does not 
    """
    outdomian = [f"train30k-{i}" for i in range(1, 27185) if f"train30k-{i}" not in all_ids]

    outdomian_dataset = {
        "safe": [],
        "unsafe": [],
    }

    for id_ in outdomian:
        try:
            is_safe = merger.sample_dict[id_]['is_safe']
        except:
            continue
        
        if is_safe == True:
            outdomian_dataset['safe'].append(id_)
        elif is_safe == False:
            outdomian_dataset['unsafe'].append(id_)
        else:
            continue
        
    return outdomian_dataset

def prepare_dataset(test_size = 10):
    """
    Prepare train dataset for later curation and the test dataset with blank filling.
    """
    concept_ids, all_ids = load_selected_concepts()
    
    # Incomplete
    train_dataset, test_dataset =  split_dataset(concept_ids)
    augment_dataset = augmentation(list(concept_ids.keys()))
    
    # Complete 
    combine_train_dataset, combine_test_dataset = combine_dataset(
        train_dataset, test_dataset, augment_dataset,
        test_size = test_size
        )
    
    # Outdomain
    outdomian_dataset = construct_outdomain(all_ids)
    
    return combine_train_dataset, combine_test_dataset, augment_dataset, outdomian_dataset

def construct_train_dataset(
    combine_train_dataset,
    outdomian_dataset,
    
    strategy='Balanced',
    
    # Random
    random_n_sample=2000,
    
    # Balanced
    balanced_samples_per_concept=20,
    
    # BalancedRatio
    balancedratio_safe_samples_per_concept=10,
    balancedratio_unsafe_samples_per_concept=10,
    
    # SafeOnlyRandom
    safeonlyrandom_n_sample=2000,
    
    # SafeOnlyBalanced
    safeonlybalanced_samples_per_concept=20,
    
    # UnsafeOnlyRandom
    unsafeonlyrandom_n_sample=2000,
    
    # UnsafeOnlyBalanced
    unsafeonlybalanced_samples_per_concept=20,
    
    # ExtraOutDomain
    extraindomain_n_sample=1000,
    extraoutdomain_n_sample=1000,

    # SafeExtraOutDomain
    safeextraindomain_n_sample=1000,
    safeextraoutdomain_n_sample=1000,
    
    ):
    """
    Augmented training dataset via different strategies.
    RETURN: List of selected dataset ids
    """
    id_safe = [id for split in combine_train_dataset.values() for id in split['safe']]
    id_unsafe = [id for split in combine_train_dataset.values() for id in split['unsafe']]
    
    id_outdomain_safe = outdomian_dataset['safe']
    id_outdomain_unsafe = outdomian_dataset['unsafe']
    
    # n_safe = [len(split['safe']) for split in combine_train_dataset.values()]
    # n_unsafe = [len(split['unsafe']) for split in combine_train_dataset.values()]
    
    random.seed(42)
    # n_sample = 1000           # for random case
    # samples_per_concept = 10   # for balanced case
    
    if strategy == "Random":
        """
        随机选则的数据集; 
        concept相关样本; 不区分(原始)safe/unsafe比例; 每个concept样本数不控制
        
        mix 1, 4, 7, 3, 1, 4, ...
        """
        dataset = random.sample(id_safe+id_unsafe, k=random_n_sample)
        
        ds = Dataset.from_list([merger.sample_dict[id] for id in dataset])
        return ds
        
    elif strategy == "Balanced":
        """
        控制concept暴露强度; 
        concept 相关样本; safe/unsafe比例一致; 每个concept 取相同数量;
        取所有concept中safe和unsafe都能满足的最小值，例如各取3条，共6条/concept
        
        safe|unsafe 3|3, 3|3, 3|3, 3|3, 3|3, 3|3, ...
        """
        dataset = []
        for c, split in combine_train_dataset.items():
            if len(split['safe']) >= int(balanced_samples_per_concept/2) and len(split['unsafe']) >= int(balanced_samples_per_concept/2):
                dataset.extend(random.sample(split['safe'], k=int(balanced_samples_per_concept/2)) + 
                                        random.sample(split['unsafe'], k=int(balanced_samples_per_concept/2)))
            else:
                print(f"Not enough samples for concept {c}, safe:{len(split['safe'])}, unsafe:{len(split['unsafe'])}")
        
        ds = Dataset.from_list([merger.sample_dict[id] for id in dataset])
        return ds
    
    elif strategy == "BalancedRatio":
        """
        控制concept暴露强度; 
        concept 相关样本; safe/unsafe比例一致; 每个concept 取相同数量;
        调整 safe|unsafe的比例
        
        safe|unsafe 6|3, 6|3, 6|3, 6|3, 6|3, 6|3, ...
        """
        dataset = []
        for c, split in combine_train_dataset.items():
            if len(split['safe']) >= int(balancedratio_safe_samples_per_concept) and len(split['unsafe']) >= int(balancedratio_unsafe_samples_per_concept):
                dataset.extend(random.sample(split['safe'], k=int(balancedratio_safe_samples_per_concept)) 
                            + random.sample(split['unsafe'], k=int(balancedratio_unsafe_samples_per_concept))) 
            else:
                print(f"Not enough samples for concept {c}, safe:{len(split['safe'])}, unsafe:{len(split['unsafe'])}")   
                
        ds = Dataset.from_list([merger.sample_dict[id] for id in dataset])
        return ds
        
    elif strategy == "SafeOnlyRandom":
        """
        concept 相关样本; is_safe=True 的部分; 
        safe 1, 4, 2, 3, 7, 6, ...
        """
        dataset = random.sample(id_safe, k=safeonlyrandom_n_sample)
        
        ds = Dataset.from_list([merger.sample_dict[id] for id in dataset])
        return ds
    
    elif strategy == "SafeOnlyBalanced":
        """
        concept 相关样本; is_safe=True 的部分; 
        safe 8, 8, 8, 8, 8, 8 ...
        """
        dataset = []
        for c, split in combine_train_dataset.items():
            if len(split['safe']) >= int(safeonlybalanced_samples_per_concept):
                dataset += random.sample(split['safe'], k=int(safeonlybalanced_samples_per_concept))
            else:
                print(f"Not enough samples for concept {c}, safe:{len(split['safe'])}, unsafe:{len(split['unsafe'])}")
        
        ds = Dataset.from_list([merger.sample_dict[id] for id in dataset])
        return ds
    
    elif strategy == "UnsafeOnlyRandom":
        """
        concept 相关样本; is_safe=False 的部分; 
        safe 1, 4, 2, 3, 7, 6, ...
        """
        dataset = random.sample(id_unsafe, k=unsafeonlyrandom_n_sample)
        
        ds = Dataset.from_list([merger.sample_dict[id] for id in dataset])
        return ds
    
    elif strategy == "UnsafeOnlyBalanced":
        """
        concept 相关样本; is_safe=False 的部分; 
        safe 8, 8, 8, 8, 8, 8 ...
        """
        dataset = []
        for c, split in combine_train_dataset.items():
            if len(split['unsafe']) >= int(unsafeonlybalanced_samples_per_concept):
                dataset += random.sample(split['unsafe'], k=int(unsafeonlybalanced_samples_per_concept))
            else:
                print(f"Not enough samples for concept {c}, safe:{len(split['safe'])}, unsafe:{len(split['unsafe'])}")
        
        ds = Dataset.from_list([merger.sample_dict[id] for id in dataset])
        return ds
        
    elif strategy == "ExtraOutDomain":
        """M1-Base 的训练集 + concept 范围外的样本"""
        dataset = random.sample(id_safe+id_unsafe, k=extraindomain_n_sample) + random.sample(id_outdomain_safe+id_outdomain_unsafe, k=extraoutdomain_n_sample)
        
        ds = Dataset.from_list([merger.sample_dict[id] for id in dataset])
        return ds
    
    elif strategy == "SafeExtraOutDomain":
        """M1-Base 的训练集 + concept 范围外的 safe 样本"""
        dataset = random.sample(id_safe, k=safeextraindomain_n_sample) + random.sample(id_outdomain_safe, k=safeextraoutdomain_n_sample)
        
        lisst = [merger.sample_dict[id] for id in dataset]
        for li in lisst:
            li['knowledge_graph'] = None
        ds = Dataset.from_list(lisst)
        return ds
        
    else:
        raise ValueError("Invalid strategy")
    
def load_sample_from_id(id):
    """
    Retrieve sample information through sample_id
    """
    return merger.sample_dict[id]
    

if __name__ == "__main__":
    
    # Prepare sample information
    merger = GraphMerger()
    stats = merger.get_stats()
    
    frequency = merger.node_importance['frequency']
    frequency = {k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
    
    # Load selected concepts
    concept_ids, all_ids = load_selected_concepts()
    
    # Split train dataset and test dataset WITH varing test dataset size.
    train_dataset, test_dataset =  split_dataset(test_size=5)
    """
    for test_size in [1,2,3,4,5]:
        train, test =  split_dataset(concept_ids, test_size=test_size)

        train_count = {}
        n_train_safe = 0
        n_train_unsafe = 0
        for c, split in train.items():
            train_count[c] = {k: len(v) for k, v in split.items()}
            n_train_safe += len(split['safe'])
            n_train_unsafe += len(split['unsafe'])

        test_count = {}
        n_test_safe = 0
        n_test_unsafe = 0
        for c, split in test.items():
            test_count[c] = {k: len(v) for k, v in split.items()}
            n_test_safe += len(split['safe'])
            n_test_unsafe += len(split['unsafe'])

        print(f"Dataset split with {test_size} safe {test_size} unsafe samples per concept within test dataset, and train dataset the rest.")    
        print(f"train {n_train_safe+n_train_unsafe}, safe:unsafe = {n_train_safe}:{n_train_unsafe}")
        print(f"test {n_test_safe+n_test_unsafe}, safe:unsafe = {n_test_safe}:{n_test_unsafe}\n")  
    """
   
    # Candidate sample index
    concept_list = list(concept_ids.keys())
    augment_dataset = augmentation(concept_list)

    """
    augment_count = {}
    n_augment_safe = 0
    n_augment_unsafe = 0
    for c, split in augment.items():
        augment_count[c] = {k: len(v) for k, v in split.items()}
        n_augment_safe += len(split['safe'])
        n_augment_unsafe += len(split['unsafe'])
    print(f"augment {n_augment_safe+n_augment_unsafe}, safe:unsafe = {n_augment_safe}:{n_augment_unsafe}")
    """
    
    # Construct dataset via different strategies
    strategy='Balanced'
    strategy='Random'
    dataset = construct_train_dataset(train_dataset, augment_dataset, strategy)
    print(len(dataset))
