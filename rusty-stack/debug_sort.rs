#![allow(unused_imports)]

use rusty_stack::core::plan::PlanItem;
use rusty_stack::core::types::ValidationTier;
use std::collections::{HashMap, HashSet};

fn debug_topological_sort(items: Vec<PlanItem>) -> Vec<PlanItem> {
    // Build dependency graph
    let mut graph: HashMap<String, Vec<String>> = HashMap::new();
    let mut reverse_graph: HashMap<String, Vec<String>> = HashMap::new();
    let mut all_nodes: HashSet<String> = HashSet::new();

    println!("Input items:");
    for item in &items {
        println!("  {} -> {:?}", item.component_id, item.dependencies);
    }

    // Initialize nodes and dependency tracking
    for item in &items {
        all_nodes.insert(item.component_id.clone());
        graph.insert(item.component_id.clone(), item.dependencies.clone());
        // For reverse graph, we want to track which nodes depend on each node
        for dep in &item.dependencies {
            reverse_graph
                .entry(dep.clone())
                .or_default()
                .push(item.component_id.clone());
        }
    }

    println!("\nDependency graph:");
    for (node, deps) in &graph {
        println!("  {} -> {:?}", node, deps);
    }

    println!("\nReverse dependency graph:");
    for (node, deps) in &reverse_graph {
        println!("  {} <- {:?}", node, deps);
    }

    // Kahn's algorithm for topological sort
    let mut sorted = Vec::new();

    // Build dependency graph (which nodes each node depends on)
    let mut graph: HashMap<String, Vec<String>> = HashMap::new();

    // Build reverse dependency graph (which nodes depend on each node)
    let mut reverse_deps: HashMap<String, Vec<String>> = HashMap::new();

    // Initialize graphs
    for item in &items {
        graph.insert(item.component_id.clone(), item.dependencies.clone());
        for dep in &item.dependencies {
            reverse_deps
                .entry(dep.clone())
                .or_default()
                .push(item.component_id.clone());
        }
    }

    // Nodes with no dependencies (nodes that don't have any dependencies listed)
    // should be in the initial queue
    let mut queue: Vec<String> = items
        .iter()
        .filter(|item| item.dependencies.is_empty())
        .map(|item| item.component_id.clone())
        .collect();

    // Build reverse dependency graph (which nodes depend on each node)
    let mut reverse_deps: HashMap<String, Vec<String>> = HashMap::new();
    for item in &items {
        for dep in &item.dependencies {
            reverse_deps
                .entry(dep.clone())
                .or_default()
                .push(item.component_id.clone());
        }
    }

    println!("\nNodes without dependencies (nodes not in reverse_deps):");
    for node in &all_nodes {
        if !reverse_deps.contains_key(node) {
            println!("  {}", node);
        }
    }

    println!("\nAll nodes: {:?}", all_nodes);
    println!(
        "Reverse deps keys: {:?}",
        reverse_deps.keys().collect::<Vec<_>>()
    );

    println!("\nInitial queue: {:?}", queue);

    while let Some(node) = queue.pop() {
        sorted.push(node.clone());
        println!("  Added {} to sorted list", node);

        if let Some(dependents) = graph.get(&node) {
            for dependent in dependents {
                let rev_deps = reverse_graph.get_mut(dependent).unwrap();
                println!(
                    "  Processing dependent {}: rev_deps before={:?}",
                    dependent, rev_deps
                );
                rev_deps.retain(|n| n != &node);
                println!("  rev_deps after={:?}", rev_deps);
                if rev_deps.is_empty() {
                    reverse_graph.remove(dependent);
                    queue.push(dependent.clone());
                    println!("  Added {} to queue", dependent);
                }
            }
        }
    }

    println!("\nFinal sorted order: {:?}", sorted);

    if sorted.len() != all_nodes.len() {
        println!(
            "Cycle detected! Missing nodes: {:?}",
            all_nodes
                .difference(&sorted.iter().cloned().collect())
                .collect::<Vec<_>>()
        );
        // Don't panic, just return the partial result for debugging
        // panic!("Dependency cycle detected in plan items");
    }

    // Reconstruct plan items in topological order
    let mut item_map: HashMap<String, PlanItem> = items
        .into_iter()
        .map(|item| (item.component_id.clone(), item))
        .collect();

    sorted
        .into_iter()
        .filter_map(|id| item_map.remove(&id))
        .collect()
}

fn main() {
    // Simple test case
    let items = vec![
        PlanItem::new(
            "pytorch",
            "2.4.0",
            "2.5.0",
            ValidationTier::Validated,
            true,
            "update pytorch",
            vec!["rocm".to_string()],
            true,
        ),
        PlanItem::new(
            "rocm",
            "7.2.0",
            "7.3.0",
            ValidationTier::Validated,
            true,
            "update rocm",
            vec![],
            true,
        ),
    ];

    let sorted = debug_topological_sort(items);
    println!("\nResult:");
    for item in sorted {
        println!("  {}", item.component_id);
    }
}
