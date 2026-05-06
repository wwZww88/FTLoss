import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm

class GraphMerger:
    def __init__(self):
        # Load BeaverTails knowledge sample
        self.files = [
            "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_0-10000.json",
            "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_10000-20000.json",
            "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_20000-27185.json",
            "/FTLoss/results/knowledge/beavertails/beavertails_30k_test_data_knowledge_0-3021.json",
            
            "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_failed_0-10000.json",
            "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_failed_10000-20000.json",
            "/FTLoss/results/knowledge/beavertails/beavertails_30k_train_data_knowledge_failed_20000-27185.json",
            "/FTLoss/results/knowledge/beavertails/beavertails_30k_test_data_knowledge_failed_0-3021.json",
            ]

        self.sample_dict = self.get_samples()

        # Merge sample that has successfully generated graph
        self.graph = nx.DiGraph()
        self.merge()
        
        # Compute the importance of the 
        # self.node_importance = self.calculate_node_importance()
        
    def get_samples(self):
        """Return dict sample_list from the knowledge file
           sample_dict = {sample_id: sample_info}
        """
        sample_dict = {}
        for file in self.files:
            with open(file, 'r') as f:
                lines = f.read().splitlines() 

            #print(f"{files[0].split('/')[-1]}: {len(lines)}")

            for line in lines:
                if 'Processing beavertails_' in line:
                    continue
                try:
                    info = eval(line.split('"""')[1])
                    kg = info['knowledge_graph']
                    if type(kg) == list:
                        kg = kg[0]
                    
                    if   "_30k_train" in file:
                        sample_id = f"train30k-{info['id']}"
                    elif "_30k_test" in file:
                        sample_id = f"test30k-{info['id']}"
                    elif "_330k_train" in file:
                        sample_id = f"train330k-{info['id']}"
                    elif "_330k_test" in file:
                        sample_id = f"test330k-{info['id']}"
                        
                    #sample_id =  f"test-{info['id']}" if "_test_" in file else f"train-{info['id']}"
                    
                    sample_dict[sample_id] = {
                        'id': sample_id,
                        'prompt': info['prompt'], 
                        'response': info['response'], 
                        'category': info['category'], 
                        'is_safe': info['is_safe'], 
                        'knowledge_graph': kg  
                    }
                except:
                    continue

        return sample_dict
        
    def merge(self):
        """Merge all sample graph"""
        for _, sample in self.sample_dict.items():
            sample_id = sample['id']
            graph = sample['knowledge_graph']
            
            if graph == None:
                continue
            else:
                try:
                    self.add_graph(graph, sample_id)
                except:
                    continue
        
    def get_all_kg(self):
        """Return list of sub knowledge graph kg_all from the knowledge file
           kg_all = [kg1, kg2, ...]
        """
        # 
        
        kg_all = []
        for file in self.files:
            with open(file, 'r') as f:
                lines = f.read().splitlines() 

            #print(f"{files[0].split('/')[-1]}: {len(lines)}")

            for line in lines[1:]:
                try:
                    kg = eval(line.split('"""')[1])['knowledge_graph']
                    if type(kg) == list:
                        kg = kg[0]
                    kg_all.append(kg)
                except:
                    continue

        return kg_all
        
    def add_graph(self, graph, sample_id):
        """Add one graph"""

        for node in graph.get('nodes', []):
            node_id = node['id']
            
            if node_id not in self.graph:
                # Add new node
                self.graph.add_node(
                    node_id,
                    type={node.get('type', '')},                      # Set
                    detailed_type={node.get('detailed_type', '')},    # Set
                    source_graphs=1,                                  # Int
                    source_sample_ids = [sample_id],                  # List
                )
            else:
                # Update node's properties
                self.graph.nodes[node_id]['type'].add(node.get('type', ''))
                self.graph.nodes[node_id]['detailed_type'].add(node.get('detailed_type', ''))
                self.graph.nodes[node_id]['source_graphs'] += 1
                self.graph.nodes[node_id]['source_sample_ids'].append(sample_id)

        for edge in graph.get('edges', []):
            from_node = edge['from']
            to_node = edge['to']
            edge_label = edge.get('label', '')
            
            if not self.graph.has_edge(from_node, to_node):
                # Add new edge
                self.graph.add_edge(
                from_node, 
                to_node, 
                label={edge_label},                  # Set  
                source_graphs=1,                     # Int
                source_sample_ids = [sample_id]      # List
            )
            else:
                # Update edge's properties
                self.graph.edges[from_node, to_node]['label'].add(edge_label)
                self.graph.edges[from_node, to_node]['source_graphs'] += 1
                self.graph.edges[from_node, to_node]['source_sample_ids'].append(sample_id)
    
    def get_merged_graph(self):
        """Merge all graphs"""
        nodes_list = []
        for node_id, node_data in self.nodes.items():
            node = {
                'id': node_id,
                'type': node_data['type'],
                'detailed_type': node_data['detailed_type'],
                'appears_in': node_data['source_graphs']
            }
            nodes_list.append(node)
        
        edges_list = []
        for edge_key, edge_data in self.edges.items():
            edge = {
                'from': edge_data['from'],
                'to': edge_data['to'],
                'label': edge_data['label'],
                'appears_in': edge_data['source_graphs']
            }
            if 'labels' in edge_data and len(set(edge_data['labels'])) > 1:
                edge['all_labels'] = list(set(edge_data['labels']))
            edges_list.append(edge)
        
        return {
            'nodes': nodes_list,
            'edges': edges_list
        }
        
    def calculate_node_importance(self, methods=['degree', 'pagerank', "eigenvector", 'frequency']):
        """Compute the nodes' importance through various metrics"""
        
        importance_scores = {}
        
        # 中心性 - 节点度占总节点数的比例，识别最"繁忙"的节点
        if 'degree' in methods:
            degree_centrality = nx.degree_centrality(self.graph)
            importance_scores['degree'] = degree_centrality
        
        # 一个节点的重要性，取决于与它相连的其他节点的重要性
        if 'eigenvector' in methods:
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
            importance_scores['eigenvector'] = eigenvector
        
        # 节点重要性，识别最有影响力的节点
        if 'pagerank' in methods:
            pagerank = nx.pagerank(self.graph)
            importance_scores['pagerank'] = pagerank
        
        # 节点在sample中出现的次数  
        if 'frequency' in methods:
            frequency = {node_id: self.graph.nodes[node_id]['source_graphs'] for node_id in list(self.graph.nodes()) 
                                                                    if 'source_graphs' in self.graph.nodes[node_id]}
            importance_scores['frequency'] = frequency
            
        return importance_scores  

    def get_stats(self):
        """Get graph basic statistics"""
        if self.graph.number_of_nodes() == 0:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'node_types': 0,
                'unique_sources': 0
            }

        node_types = set()
        detailed_types = set()
        
        for _, node_data in self.graph.nodes(data=True):
            if 'type' in node_data:
                for t in node_data['type']:
                    if t:  # 非空字符串
                        node_types.add(t)
            
            if 'detailed_type' in node_data:
                for dt in node_data['detailed_type']:
                    if dt:
                        detailed_types.add(dt)

        all_source_samples = set()
        for _, node_data in self.graph.nodes(data=True):
            if 'source_sample_ids' in node_data:
                all_source_samples.update(node_data['source_sample_ids'])

        node_frequencies = []
        for _, node_data in self.graph.nodes(data=True):
            freq = node_data.get('source_graphs', 1)
            node_frequencies.append(freq)

        edge_frequencies = []
        for _, _, edge_data in self.graph.edges(data=True):
            freq = edge_data.get('source_graphs', 1)
            edge_frequencies.append(freq)

        avg_degree = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        
        return {
            'basic_stats': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'node_types_count': len(node_types),
                'detailed_types_count': len(detailed_types),
                'unique_sources_count': len(all_source_samples),
                'avg_degree': avg_degree,
                'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
            },
            'node_stats': {
                'unique_node_types': list(node_types)[:20],  # 前20个
                'node_frequency_stats': {
                    'min': min(node_frequencies) if node_frequencies else 0,
                    'max': max(node_frequencies) if node_frequencies else 0,
                    'avg': sum(node_frequencies) / len(node_frequencies) if node_frequencies else 0,
                    'total': sum(node_frequencies)
                },
                'most_frequent_nodes': self.get_most_frequent_nodes(10)
            },
            'edge_stats': {
                'edge_frequency_stats': {
                    'min': min(edge_frequencies) if edge_frequencies else 0,
                    'max': max(edge_frequencies) if edge_frequencies else 0,
                    'avg': sum(edge_frequencies) / len(edge_frequencies) if edge_frequencies else 0,
                    'total': sum(edge_frequencies)
                },
                'unique_labels_count': self.get_unique_edge_labels_count(),
                'most_frequent_edges': self.get_most_frequent_edges(10)
            },
            'graph_stats': {
                'is_directed': self.graph.is_directed(),
                'is_multigraph': self.graph.is_multigraph(),
                'number_of_connected_components': self.get_connected_components_count(),
                'average_clustering': nx.average_clustering(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else 0
            }
        }

    def get_most_frequent_nodes(self, top_n=10):
        """获取出现频率最高的节点"""
        nodes_with_freq = []
        for node_id, node_data in self.graph.nodes(data=True):
            freq = node_data.get('source_graphs', 1)
            nodes_with_freq.append({
                'node_id': node_id,
                'frequency': freq,
                'type': list(node_data.get('type', {''}))[0] if node_data.get('type') else '',
                'num_samples': len(node_data.get('source_sample_ids', []))
            })
        
        nodes_with_freq.sort(key=lambda x: x['frequency'], reverse=True)
        return nodes_with_freq[:top_n]

    def get_most_frequent_edges(self, top_n=10):
        """获取出现频率最高的边"""
        edges_with_freq = []
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            freq = edge_data.get('source_graphs', 1)
            edges_with_freq.append({
                'from': from_node,
                'to': to_node,
                'frequency': freq,
                'label': list(edge_data.get('label', {''}))[0] if edge_data.get('label') else '',
                'num_samples': len(edge_data.get('source_sample_ids', []))
            })
        
        edges_with_freq.sort(key=lambda x: x['frequency'], reverse=True)
        return edges_with_freq[:top_n]

    def get_unique_edge_labels_count(self):
        """统计不同的边标签数量"""
        unique_labels = set()
        for _, _, edge_data in self.graph.edges(data=True):
            if 'label' in edge_data:
                for label in edge_data['label']:
                    if label:
                        unique_labels.add(label)
        
        return len(unique_labels)

    def get_connected_components_count(self):
        """获取连通分量数量（转换为无向图）"""
        if self.graph.number_of_nodes() == 0:
            return 0
        
        undirected_graph = self.graph.to_undirected()
        return nx.number_connected_components(undirected_graph)
    
def find_important_nodes(degree_dict, pagerank_dict, eigenvector_dict, top_k=1000):
    """
    找出三个度量下都重要的节点
    
    Args:
        degree_dict: 度中心性字典
        pagerank_dict: PageRank字典
        eigenvector_dict: 特征向量中心性字典
        top_k: 每个度量下取前多少个节点
    
    Returns:
        包含结果的字典
    """
    # 1. 验证输入
    print(f"度中心性节点数: {len(degree_dict)}")
    print(f"PageRank节点数: {len(pagerank_dict)}")
    print(f"特征向量中心性节点数: {len(eigenvector_dict)}")
    
    # 检查是否有共同的节点
    all_nodes = set(degree_dict.keys()) | set(pagerank_dict.keys()) | set(eigenvector_dict.keys())
    print(f"总节点数: {len(all_nodes)}")
    
    # 2. 找出每个度量下的top_k节点
    print(f"\n找出每个度量下的前{top_k}个节点...")
    
    # 度中心性top_k
    degree_top = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    degree_top_nodes = set(node for node, _ in degree_top)
    print(f"度中心性前{top_k}节点获取完成")
    
    # PageRank top_k
    pagerank_top = sorted(pagerank_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    pagerank_top_nodes = set(node for node, _ in pagerank_top)
    print(f"PageRank前{top_k}节点获取完成")
    
    # 特征向量中心性top_k
    eigenvector_top = sorted(eigenvector_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    eigenvector_top_nodes = set(node for node, _ in eigenvector_top)
    print(f"特征向量中心性前{top_k}节点获取完成")
    
    # 3. 找出交集
    
    # 两两交集
    degree_pagerank_intersection = degree_top_nodes & pagerank_top_nodes
    degree_eigenvector_intersection = degree_top_nodes & eigenvector_top_nodes
    pagerank_eigenvector_intersection = pagerank_top_nodes & eigenvector_top_nodes
    
    # 三个度量的交集
    triple_intersection = degree_top_nodes & pagerank_top_nodes & eigenvector_top_nodes
    
    # 至少出现在两个度量中的节点
    at_least_two = set()
    node_counts = defaultdict(int)
    
    for node in degree_top_nodes:
        node_counts[node] += 1
    for node in pagerank_top_nodes:
        node_counts[node] += 1
    for node in eigenvector_top_nodes:
        node_counts[node] += 1
    
    for node, count in node_counts.items():
        if count >= 2:
            at_least_two.add(node)
    
    # 4. 收集结果
    results = {
        'degree_top': dict(degree_top),  # 前top_k节点的完整信息
        'pagerank_top': dict(pagerank_top),
        'eigenvector_top': dict(eigenvector_top),
        'degree_top_nodes': degree_top_nodes,
        'pagerank_top_nodes': pagerank_top_nodes,
        'eigenvector_top_nodes': eigenvector_top_nodes,
        'degree_pagerank_intersection': degree_pagerank_intersection,
        'degree_eigenvector_intersection': degree_eigenvector_intersection,
        'pagerank_eigenvector_intersection': pagerank_eigenvector_intersection,
        'triple_intersection': triple_intersection,
        'at_least_two': at_least_two,
        'node_counts': node_counts  # 节点出现在几个度量中
    }
    
    return results

def analyze_intersection_results(results):
    """分析交集结果"""
    print("=" * 60)
    print("重要节点分析结果")
    print("=" * 60)
    
    print(f"\n1. 各度量前1000节点统计:")
    print(f"   度中心性前1000节点数: {len(results['degree_top_nodes'])}")
    print(f"   PageRank前1000节点数: {len(results['pagerank_top_nodes'])}")
    print(f"   特征向量中心性前1000节点数: {len(results['eigenvector_top_nodes'])}")
    
    print(f"\n2. 节点重合情况:")
    print(f"   度中心性 ∩ PageRank: {len(results['degree_pagerank_intersection'])} 个节点")
    print(f"   度中心性 ∩ 特征向量: {len(results['degree_eigenvector_intersection'])} 个节点")
    print(f"   PageRank ∩ 特征向量: {len(results['pagerank_eigenvector_intersection'])} 个节点")
    print(f"   三者交集: {len(results['triple_intersection'])} 个节点")
    print(f"   至少出现在两个度量中: {len(results['at_least_two'])} 个节点")
    
    print(f"\n3. 节点出现频率分布:")
    freq_dist = defaultdict(int)
    for node, count in results['node_counts'].items():
        freq_dist[count] += 1
    
    for freq, num_nodes in sorted(freq_dist.items()):
        print(f"   出现在 {freq} 个度量中: {num_nodes} 个节点")
    
    return freq_dist

def get_detailed_intersection_info(results, top_n=50, print_=False):
    """获取交集的详细信息"""
    
    print(f"\n4. 三个度量都重要的前{top_n}个节点:")
    print("-" * 80)
    
    # 获取三个度量都包含的节点
    triple_nodes = list(results['triple_intersection'])
    
    if triple_nodes:
        # 计算综合得分
        node_scores = []
        for node in triple_nodes:
            # 标准化每个度量下的排名得分
            degree_score = 1.0
            pagerank_score = 1.0
            eigenvector_score = 1.0
            
            # 计算综合得分
            composite_score = (degree_score + pagerank_score + eigenvector_score) / 3.0
            node_scores.append((node, composite_score))
        
        # 按综合得分排序
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        if print_:
            print(f"排名 | {'节点ID':<40} | 度中心性 | PageRank | 特征向量 | 综合分")
            print("-" * 80)
            
            for i, (node, composite) in enumerate(node_scores[:top_n], 1):
                degree_val = results['degree_top'].get(node, 0)
                pagerank_val = results['pagerank_top'].get(node, 0)
                eigenvector_val = results['eigenvector_top'].get(node, 0)
                
                print(f"{i:4d} | {node[:40]:<40} | {degree_val:.6f} | {pagerank_val:.6f} | {eigenvector_val:.6f} | {composite:.6f}")
    else:
        print("没有节点同时出现在三个度量的前1000名中")
    
    return triple_nodes[:top_n] if triple_nodes else []

def export_important_nodes(results, output_file='important_nodes.json'):
    """导出重要节点到文件"""
    import json
    
    # 准备导出数据
    export_data = {
        'triple_intersection': {
            'count': len(results['triple_intersection']),
            'nodes': list(results['triple_intersection'])
        },
        'at_least_two_metrics': {
            'count': len(results['at_least_two']),
            'nodes': list(results['at_least_two'])
        },
        'node_details': {}
    }
    
    # 添加节点详细信息
    for node in results['triple_intersection']:
        export_data['node_details'][node] = {
            'degree_centrality': results['degree_top'].get(node, 0),
            'pagerank': results['pagerank_top'].get(node, 0),
            'eigenvector_centrality': results['eigenvector_top'].get(node, 0),
            'appears_in_metrics': results['node_counts'].get(node, 0)
        }
    
    # 导出到JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已导出到: {output_file}")
    return export_data

def calculate_consistency_scores(results):
    """计算度量间的一致性得分"""
    
    print("\n5. 度量间一致性分析:")
    print("-" * 60)
    
    # Jaccard相似系数
    def jaccard_similarity(set1, set2):
        if len(set1 | set2) == 0:
            return 0
        return len(set1 & set2) / len(set1 | set2)
    
    # 计算各种Jaccard相似度
    jaccard_degree_pagerank = jaccard_similarity(
        results['degree_top_nodes'], 
        results['pagerank_top_nodes']
    )
    
    jaccard_degree_eigenvector = jaccard_similarity(
        results['degree_top_nodes'], 
        results['eigenvector_top_nodes']
    )
    
    jaccard_pagerank_eigenvector = jaccard_similarity(
        results['pagerank_top_nodes'], 
        results['eigenvector_top_nodes']
    )
    
    print(f"度中心性 vs PageRank Jaccard相似度: {jaccard_degree_pagerank:.3%}")
    print(f"度中心性 vs 特征向量 Jaccard相似度: {jaccard_degree_eigenvector:.3%}")
    print(f"PageRank vs 特征向量 Jaccard相似度: {jaccard_pagerank_eigenvector:.3%}")

    print(f"\n重叠率（从每个度量的视角）:")
    print(f"度中心性节点中，也出现在PageRank前1000的比例: {len(results['degree_pagerank_intersection'])/1000:.1%}")
    print(f"度中心性节点中，也出现在特征向量前1000的比例: {len(results['degree_eigenvector_intersection'])/1000:.1%}")
    print(f"PageRank节点中，也出现在度中心性前1000的比例: {len(results['degree_pagerank_intersection'])/1000:.1%}")
    
    return {
        'jaccard_degree_pagerank': jaccard_degree_pagerank,
        'jaccard_degree_eigenvector': jaccard_degree_eigenvector,
        'jaccard_pagerank_eigenvector': jaccard_pagerank_eigenvector
    }
    
def get_induced_subgraph(graph, selected_nodes):
    """
    获取由选定节点诱导的子图
    只包含这些节点，以及它们之间的边
    
    Args:
        graph: NetworkX图对象
        selected_nodes: 选定的节点列表
    
    Returns:
        诱导子图
    """
    # 过滤存在的节点
    existing_nodes = [node for node in selected_nodes if node in graph]
    missing_nodes = [node for node in selected_nodes if node not in graph]
    
    if missing_nodes:
        print(f"警告: 以下节点不在图中: {missing_nodes}")
        print(f"找到 {len(existing_nodes)}/{len(selected_nodes)} 个节点")
    
    if not existing_nodes:
        print("错误: 没有找到任何节点在图中")
        return None
    
    # 创建诱导子图
    subgraph = graph.subgraph(existing_nodes)
    
    print(f"诱导子图信息:")
    print(f"  节点数: {subgraph.number_of_nodes()}")
    print(f"  边数: {subgraph.number_of_edges()}")
    print(f"  密度: {nx.density(subgraph):.4f}")
    
    return subgraph

def get_extended_subgraph(graph, selected_nodes, depth=1, include_all_edges=True):
    """
    获取扩展子图，包含选定节点及其邻居
    
    Args:
        graph: NetworkX图对象
        selected_nodes: 选定的节点列表
        depth: 扩展深度（1: 直接邻居，2: 两跳邻居等）
        include_all_edges: 是否包含邻居之间的边
    
    Returns:
        扩展子图
    """

    existing_nodes = [node for node in selected_nodes if node in graph]
    
    if not existing_nodes:
        print("错误: 没有找到任何节点在图中")
        return None

    all_nodes = set(existing_nodes)
    
    # 扩展到指定深度的邻居
    for _ in range(depth):
        new_neighbors = set()
        for node in all_nodes:
            if node in graph:
                # 添加所有邻居
                new_neighbors.update(graph.neighbors(node))
                new_neighbors.update(graph.predecessors(node))
        
        all_nodes.update(new_neighbors)

    if include_all_edges:
        subgraph = graph.subgraph(all_nodes)
    else:
        # 只保留与选定节点相关的边
        subgraph = nx.DiGraph()

        for node in all_nodes:
            if node in graph:
                subgraph.add_node(node, **graph.nodes[node])
        
        # 添加与选定节点相关的边
        for u, v, data in graph.edges(data=True):
            if u in all_nodes and v in all_nodes:
                # 至少一端是原始选定节点
                if u in existing_nodes or v in existing_nodes:
                    subgraph.add_edge(u, v, **data)
                    
    new_nodes = [node for node in subgraph if node not in existing_nodes]
        
    
    print(f"扩展子图信息 (深度={depth}):")
    print(f"  总节点数: {subgraph.number_of_nodes()}")
    print(f"  原始选定节点: {len(existing_nodes)}")
    print(f"  额外节点: {subgraph.number_of_nodes() - len(existing_nodes)}")
    print(f"  边数: {subgraph.number_of_edges()}")
    print(f"  密度: {nx.density(subgraph):.4f}")
    
    return subgraph, new_nodes, list(existing_nodes)

if __name__ == "__main__":
    # Get the knowledge graph from local file
    merger = GraphMerger()
    stats = merger.get_stats()
    
    # Compute the importance of knowledge node
    degree_centrality = merger.node_importance['degree']
    pagerank = merger.node_importance['pagerank']
    eigenvector = merger.node_importance['eigenvector']
    frequency = merger.node_importance['frequency']
    print(len(degree_centrality), len(pagerank), len(eigenvector), len(frequency))

    results = find_important_nodes(
    degree_dict=degree_centrality,
    pagerank_dict=pagerank,
    eigenvector_dict=eigenvector,
    top_k=1000
    )
    
    # Get the most important nodes
    freq_dist = analyze_intersection_results(results)
    top_nodes = get_detailed_intersection_info(results, top_n=200)
    consistency = calculate_consistency_scores(results)
    
    # Get subgraph from selected nodes
    selected_nodes = [
    'weed', 'target', 'insult', 'bag', 'face', 'criminals', 
    'clothes', 'location', 'kid', 'blacks', 'respect', 'heart', 
    'humans', 'urine', 'laptop', 'parents', 'lock', 'Muslims', 
    'Walmart', 'ugly people'
    ]
    subgraph = get_induced_subgraph(merger.graph, selected_nodes)
    
    subgraph_depth_1, new_nodes_depth_1, original_nodes = get_extended_subgraph(merger.graph, selected_nodes, depth=1)
    subgraph_depth_2, new_nodes_depth_2, original_nodes = get_extended_subgraph(merger.graph, selected_nodes, depth=2)
    
    # Find sample information from node
    node = 'stabbing'
    source_sample_ids = subgraph_depth_1.nodes[node]['source_sample_ids']
    sample_id = source_sample_ids[-1]
    print(merger.sample_dict[sample_id])
